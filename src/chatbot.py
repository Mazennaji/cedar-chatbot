import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.normalizer import ArabiziNormalizer
from src.language_detector import LanguageDetector, Language
from src.intent_classifier import IntentClassifier, Intent
from src.sentiment import SentimentAnalyzer
from src.memory import ConversationMemory
from src.generator import MultilingualGenerator

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    response: str
    session_id: str
    message_id: str = ""
    metadata: dict = field(default_factory=dict)


IDENTITY_PATTERNS_EN = (
    "what is your name", "what's your name", "whats your name",
    "what your name", "do you have a name", "your name is",
    "your name", "tell me your name", "may i know your name",
    "who are you", "what are you", "introduce yourself",
    "tell me about yourself", "what should i call you",
    "what do they call you",
    "do you have pets", "do you have a pet", "do you have any pets",
    "do you have a family", "do you have kids", "do you have children",
    "do you have siblings", "do you have a brother", "do you have a sister",
    "where do you live", "where are you from", "where were you born",
    "how old are you", "what's your age", "whats your age", "your age",
    "what do you do for a living", "what's your job", "whats your job",
    "are you human", "are you a robot", "are you an ai", "are you a bot",
    "are you real", "are you a person",
)

IDENTITY_PATTERNS_AR = (
    "ما اسمك", "ما هو اسمك", "ما إسمك", "شو اسمك", "شو إسمك",
    "اسمك", "إسمك", "ما اسم", "كيف نناديك", "شو بنناديك",
    "من أنت", "من انت", "مين انت", "مين أنت",
    "عرف نفسك", "عرّف نفسك", "من تكون", "حدثني عن نفسك",
    "عندك أولاد", "عندك ولاد", "عندك عيلة", "عندك عائلة",
    "عندك حيوان", "عندك حيوانات", "عندك أخوات", "عندك إخوة",
    "وين ساكن", "من وين انت", "من أين أنت", "أين تعيش",
    "كم عمرك", "شو عمرك", "عمرك",
    "أنت إنسان", "هل أنت إنسان", "أنت روبوت", "هل أنت روبوت",
    "أنت ذكاء اصطناعي", "هل أنت بشر",
)

IDENTITY_PATTERNS_ARABIZI = (
    "shu esmak", "shu ismak", "esmak shu", "ismak shu",
    "shu esmek", "shu ismek", "esmek shu", "ismek shu",
    "esmak", "ismak", "esmek", "ismek",
    "min enta", "min inta", "min ente", "min inte",
    "3arrefne 3a 7alak", "3arrefne", "3arrefna",
    "7akine 3an 7alak", "min hayda",
    "3andak wled", "3andak welad", "3andak 3ayle", "3andak 3ayleh",
    "3andak 7ayawan", "3andak 7ayawanet", "3andak ekhwe",
    "wen sakin", "min wen enta", "min wen inta",
    "kam 3omrak", "kam 3omrek", "shu 3omrak", "3omrak",
    "enta insan", "enta robot", "enta bot", "enta ai", "enta zaka2",
    "are you human", "are you a bot", "are you a robot",
)


IDENTITY_RESPONSE_EN = (
    "I'm Cedar 🌲 — a trilingual AI assistant. I understand English, "
    "Modern Standard Arabic, and Lebanese dialect including Arabizi. "
    "I don't have personal experiences like pets, a family, or a hometown, "
    "but I'm here to help with questions about AI, NLP, and language. "
    "What would you like to talk about?"
)

IDENTITY_RESPONSE_AR = (
    "أنا سيدر 🌲 — مساعد ذكاء اصطناعي ثلاثي اللغات. "
    "أفهم الإنجليزية، والعربية الفصحى، واللهجة اللبنانية بما فيها الأرابيزي. "
    "ما عندي تجارب شخصية متل حيوانات أو عيلة أو مسقط رأس، "
    "بس موجود لساعدك بأسئلة عن الذكاء الاصطناعي ومعالجة اللغة. "
    "شو حابب نحكي عنه؟"
)

IDENTITY_RESPONSE_ARABIZI = (
    "Ana Cedar 🌲 — assistant zaki bi tlet lughat. "
    "Bif7am inglizi, 3arabi fos7a, w lubnani 7atta el arabizi. "
    "Ma 3ande tajaroub shakhsiyye mitl 7ayawanet aw 3ayle aw balad, "
    "bas ana hon la sa3dak bi as2ile 3an AI w el lugha. "
    "Shu baddak na7ke 3anno?"
)


KNOWLEDGE_BASE = {
    "what is nlp": (
        "NLP stands for Natural Language Processing — a field of AI that helps computers "
        "understand, interpret, and generate human language. It powers things like chatbots, "
        "translation, speech recognition, and sentiment analysis."
    ),
    "what is natural language processing": (
        "Natural Language Processing (NLP) is a branch of artificial intelligence focused on "
        "enabling computers to understand and work with human language. It includes tasks like "
        "text classification, machine translation, named entity recognition, and more."
    ),
    "what is ai": (
        "AI (Artificial Intelligence) is the simulation of human intelligence by machines. "
        "It includes subfields like machine learning, deep learning, NLP, and computer vision."
    ),
    "what is artificial intelligence": (
        "Artificial Intelligence is the science of building machines that can perform tasks "
        "that normally require human intelligence — like reasoning, learning, and understanding language."
    ),
    "what is ml": (
        "ML stands for Machine Learning — a subset of AI where models learn patterns from data "
        "instead of being explicitly programmed. Examples include decision trees, neural networks, "
        "and support vector machines."
    ),
    "what is machine learning": (
        "Machine Learning is a branch of AI that allows systems to learn from data and improve "
        "over time without being explicitly programmed. It's used in recommendations, fraud detection, "
        "image recognition, and much more."
    ),
    "what is deep learning": (
        "Deep Learning is a subset of machine learning that uses neural networks with many layers "
        "to learn from large amounts of data. It powers image recognition, speech synthesis, and LLMs."
    ),
    "what is a neural network": (
        "A neural network is a computational model inspired by the human brain. It consists of "
        "layers of interconnected nodes (neurons) that learn to recognize patterns from data."
    ),
    "what is a large language model": (
        "A Large Language Model (LLM) is a deep learning model trained on massive text datasets "
        "to understand and generate human language. Examples include GPT-4, Claude, and Gemini."
    ),
    "what is llm": (
        "LLM stands for Large Language Model — an AI model trained on vast amounts of text to "
        "understand and generate language. Examples: GPT-4, Claude, Gemini, LLaMA."
    ),
    "what is a transformer": (
        "A Transformer is a neural network architecture introduced in 2017 that uses self-attention "
        "to process sequences in parallel. It is the foundation of modern LLMs like BERT and GPT."
    ),
    "what is bert": (
        "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained NLP model "
        "by Google. It reads text bidirectionally and is widely used for classification and Q&A."
    ),
    "what is gpt": (
        "GPT (Generative Pre-trained Transformer) is a family of large language models by OpenAI. "
        "GPT models generate human-like text and are the basis for ChatGPT."
    ),
    "what is cedar": (
        "Cedar is this trilingual AI chatbot! It supports English, Modern Standard Arabic, and "
        "Lebanese dialect including Arabizi. It uses BlenderBot for English generation, mT5 for "
        "Arabic and Arabizi generation, and custom NLP modules for language detection, "
        "intent classification, and sentiment analysis."
    ),
    "tell me about lebanon": (
        "Lebanon is a small country in the Middle East on the Mediterranean coast. It's known for "
        "its rich history, diverse culture, cedar forests (the national symbol 🌲), and cuisine "
        "like hummus, tabbouleh, and kibbeh. Beirut is its capital."
    ),
    "what is arabizi": (
        "Arabizi (also called Franco-Arabic) is Arabic written in Latin script with numbers "
        "replacing sounds that don't exist in English. For example: 7 = ح, 3 = ع, 2 = ء. "
        "It's widely used by Lebanese and Arab youth online."
    ),
    "what is sentiment analysis": (
        "Sentiment analysis is an NLP task that determines the emotional tone of text — whether "
        "it's positive, negative, or neutral. It's used in product reviews, social media monitoring, "
        "and customer feedback systems."
    ),
    "what is intent classification": (
        "Intent classification is an NLP task that identifies what a user wants to accomplish from "
        "their message — like greeting, asking a question, or making a request. It's core to "
        "chatbot and virtual assistant systems."
    ),
    "what is named entity recognition": (
        "Named Entity Recognition (NER) is an NLP task that identifies and classifies named entities "
        "in text — such as people, organizations, locations, and dates."
    ),
    "what is ner": (
        "NER stands for Named Entity Recognition — an NLP technique that detects names, places, "
        "organizations, and other entities in text."
    ),
    "what is tokenization": (
        "Tokenization is the process of splitting text into smaller units called tokens — usually "
        "words or subwords. It's one of the first steps in any NLP pipeline."
    ),
    "what is a chatbot": (
        "A chatbot is a software application that simulates human conversation through text or voice. "
        "Modern chatbots use NLP and machine learning to understand context and generate responses."
    ),
    "what is blenderbot": (
        "BlenderBot is an open-domain chatbot by Meta AI, trained on large conversational datasets. "
        "Cedar uses BlenderBot-400M-distill as its base conversation engine for English responses."
    ),
}

ARABIZI_KNOWLEDGE_BASE = {
    "what is nlp": (
        "NLP ya3ne Natural Language Processing — ya3ne kif el computer bif7am el lugha el bashariyye. "
        "Bi sha8el el chatbots, el tarjame, w tawzif el masha3er. Ktir muhim la AI!"
    ),
    "what is ai": (
        "AI ya3ne Artificial Intelligence — ya3ne el zaka2 el istina3i. El ma7ine btit3allem w btfakker "
        "mitl el insan! Bi sha8el fi ChatGPT, el sayyarat el dhatiyye, w ktir akyad tane."
    ),
    "what is machine learning": (
        "Machine Learning ya3ne el ma7ine btit3allem min el data la7ala, bidoun ma 7ada ybarmjha. "
        "Mitlan, Netflix byit3allem shu baddak tshuf min 5ilal shu shi7to abel — 7elo, ma?"
    ),
    "what is deep learning": (
        "Deep Learning hiyye teknik bi AI btusted3mil shabakat 3asabiyye ktir tawile la tit3allem. "
        "Hayde shi8liyyet ChatGPT w el tawlid el sawti. Ktir 2awwiye!"
    ),
    "what is a neural network": (
        "Shabake 3asabiyye hiyye model 7asubi mstaw7a min el dimagh el bashari. Fiha tbaqa2at min "
        "3u2ad mrabbouta bi ba3da btit3allem tit3arraf 3al anamit min el data."
    ),
    "what is llm": (
        "LLM ya3ne Large Language Model — model AI mit3allem 3a kammiyyat dakhme min el nusus. "
        "Mitlan GPT-4, Claude, w Gemini. Hayde el hayawiyyat bi sha8ilo el chatbots el 7diliye!"
    ),
    "what is a transformer": (
        "Transformer hiyye binye shabake 3asabiyye ija bi 2017 btusted3mil 'self-attention' la t3abbi "
        "el nusus bi parallel. Hayde el asas ta7t kell nmadhaj LLM mitl BERT w GPT."
    ),
    "what is bert": (
        "BERT model NLP mn Google — byiqra el nass min l-jihatain (msh bass min shmal la ymin). "
        "Ktir mnesta3melo la tassif el nusus w el as2ile w el ajwibe."
    ),
    "what is gpt": (
        "GPT hiyye 3a2ile nmadhaj LLM mn OpenAI. ChatGPT mabni 3ala GPT. Btawwil nusus "
        "tib2a zayy ma katabha el insan — ktir sha6er!"
    ),
    "what is cedar": (
        "Cedar huwwe ana! Chatbot btit7akka ma3i bil inglizi, 3arabi, w arabizi. "
        "3amil bi BlenderBot la inglizi, mT5 la 3arabi w arabizi, w modules NLP. 🌲"
    ),
    "tell me about lebanon": (
        "Lebnen balad zghir 3a sa7el el ba7r el abyad el mutawassi6, ktir 7elo! Ma3ruf bi tariko "
        "el 3ariq, el akl el laziz mitl hummus w tabbule w kibbe, w 3abel el arz 🌲. Beirut hiyye 3asimto."
    ),
    "what is arabizi": (
        "Arabizi — aw Franco-Arabic — hiyye el 3arabi maktub bi 7orouf latiniyye ma3 ar2am btistabidd "
        "el as7wat. Mitlan: 7 = ح, 3 = ع, 2 = ء. Ktir musta3male 3and el shebab el lubnani!"
    ),
    "what is sentiment analysis": (
        "Tawzif el masha3er ya3ne el computer bif7am iza el nass ijabi aw salabi aw me7yad. "
        "Bista3melou la 2ira2et taqyimat el muntajat w muraqabet el social media."
    ),
    "what is a chatbot": (
        "Chatbot huwwe barnamaj byid3i yit7akka ma3ak mitl el insan — ana mitlan! El chatbots "
        "el 7diliyye bitusted3mil AI la yif7amu el siyaq w yjawibu b6arike mna3be."
    ),
    "what is tokenization": (
        "Tokenization ya3ne taqsim el nass la wa7idat as8ar esma tokens — 3adatan kalimat aw ajza2 "
        "min kalimat. Hayde min awwal l-5u6ouat bi ay pipeline NLP."
    ),
}

ARABIC_KNOWLEDGE_BASE = {
    "ما هو nlp": (
        "NLP تعني معالجة اللغة الطبيعية — وهي مجال في الذكاء الاصطناعي يساعد الحواسيب على "
        "فهم اللغة البشرية وتفسيرها وتوليدها. تُستخدم في الترجمة، وتحليل المشاعر، والمساعدين الذكيين."
    ),
    "ما هي معالجة اللغة الطبيعية": (
        "معالجة اللغة الطبيعية (NLP) هي فرع من الذكاء الاصطناعي يُمكّن الحواسيب من فهم اللغة البشرية "
        "والعمل معها. تشمل مهام مثل تصنيف النصوص، والترجمة الآلية، والتعرف على الكيانات المسماة."
    ),
    "ما هو الذكاء الاصطناعي": (
        "الذكاء الاصطناعي (AI) هو محاكاة الذكاء البشري بواسطة الآلات. "
        "يشمل مجالات فرعية مثل تعلم الآلة، والتعلم العميق، ومعالجة اللغة الطبيعية، ورؤية الحاسوب."
    ),
    "اشرح لي الذكاء الاصطناعي": (
        "الذكاء الاصطناعي هو علم بناء أنظمة قادرة على أداء مهام تتطلب عادةً ذكاءً بشرياً، "
        "مثل التفكير والتعلم وفهم اللغة. من أبرز تطبيقاته: المساعدون الصوتيون، وسيارات القيادة الذاتية، "
        "وأنظمة التوصية."
    ),
    "ما هو تعلم الآلة": (
        "تعلم الآلة هو فرع من الذكاء الاصطناعي يتعلم فيه النموذج الأنماط من البيانات بدلاً من "
        "البرمجة الصريحة. يُستخدم في التوصيات، واكتشاف الاحتيال، والتعرف على الصور."
    ),
    "ما هو التعلم الآلي": (
        "التعلم الآلي هو قدرة الأنظمة على التعلم من البيانات والتحسن مع الوقت تلقائياً. "
        "يُستخدم في مجالات عديدة مثل التشخيص الطبي، وتصفية البريد العشوائي، والترجمة الآلية."
    ),
    "ما هو التعلم العميق": (
        "التعلم العميق هو فرع من تعلم الآلة يستخدم شبكات عصبية متعددة الطبقات لتعلم أنماط "
        "معقدة من كميات ضخمة من البيانات. يُشغّل التعرف على الصور، وتوليد الكلام، والنماذج اللغوية الكبيرة."
    ),
    "اشرح لي التعلم العميق": (
        "التعلم العميق هو تقنية ذكاء اصطناعي تعتمد على شبكات عصبية اصطناعية ذات طبقات متعددة. "
        "تُعلَّم هذه الشبكات على كميات ضخمة من البيانات لتتعرف على الأنماط. "
        "تُستخدم في تطبيقات مثل ChatGPT، وتشخيص الأمراض من الصور الطبية، والسيارات ذاتية القيادة."
    ),
    "ما هي الشبكة العصبية": (
        "الشبكة العصبية هي نموذج حسابي مستوحى من الدماغ البشري. تتكون من طبقات من العقد المترابطة "
        "(الخلايا العصبية الاصطناعية) التي تتعلم التعرف على الأنماط من البيانات."
    ),
    "ما هو نموذج اللغة الكبير": (
        "نموذج اللغة الكبير (LLM) هو نموذج تعلم عميق مُدرَّب على كميات ضخمة من النصوص "
        "لفهم اللغة البشرية وتوليدها. من أبرز الأمثلة: GPT-4، وكلود، وجيميني."
    ),
    "ما هو المحوّل": (
        "المحوّل (Transformer) هو بنية شبكة عصبية ظهرت عام 2017 تستخدم آلية الانتباه الذاتي "
        "لمعالجة التسلسلات بشكل متوازٍ. وهو الأساس الذي تبنى عليه النماذج الحديثة مثل BERT وGPT."
    ),
    "ما هو تحليل المشاعر": (
        "تحليل المشاعر هو مهمة في NLP تحدد النبرة العاطفية للنص — إيجابية أو سلبية أو محايدة. "
        "يُستخدم في مراجعات المنتجات، ومراقبة وسائل التواصل الاجتماعي، وتحليل آراء العملاء."
    ),
    "ما هو تصنيف النية": (
        "تصنيف النية هو مهمة NLP تحدد ما يريد المستخدم تحقيقه من رسالته — "
        "كالترحيب، أو طرح سؤال، أو تقديم طلب. وهو أساسي في أنظمة المساعدين الافتراضيين."
    ),
    "ما هو الترميز": (
        "الترميز هو عملية تقسيم النص إلى وحدات أصغر تسمى tokens — عادةً كلمات أو أجزاء من كلمات. "
        "وهو من الخطوات الأولى في أي خط أنابيب لمعالجة اللغة الطبيعية."
    ),
    "ما هو روبوت الدردشة": (
        "روبوت الدردشة هو تطبيق برمجي يحاكي المحادثة البشرية عبر النص أو الصوت. "
        "يستخدم الروبوتات الحديثة NLP وتعلم الآلة لفهم السياق وتوليد ردود مناسبة."
    ),
    "ما هو سيدار": (
        "سيدار هو هذا الروبوت المحادثة ثلاثي اللغات! يدعم اللغة الإنجليزية، والعربية الفصحى، "
        "واللهجة اللبنانية بما فيها الأرابيزي. يستخدم BlenderBot للإنجليزية، وmT5 للعربية والأرابيزي، "
        "ووحدات NLP مخصصة للكشف عن اللغة، وتصنيف النية، وتحليل المشاعر."
    ),
    "أخبرني عن لبنان": (
        "لبنان بلد صغير في الشرق الأوسط على ساحل البحر الأبيض المتوسط. يشتهر بتاريخه العريق، "
        "وثقافته المتنوعة، وغاباته من أشجار الأرز 🌲، ومطبخه الشهير كالحمص والتبولة والكبة. "
        "بيروت هي عاصمته."
    ),
    "ما هي الأرابيزي": (
        "الأرابيزي هي العربية المكتوبة بحروف لاتينية مع أرقام تحل محل أصوات غير موجودة في الإنجليزية. "
        "مثلاً: 7 = ح، 3 = ع، 2 = ء. يستخدمها الشباب اللبناني والعربي على الإنترنت."
    ),
}


_AR_KB_KEYWORD_MAP = [
    (["nlp", "معالجة اللغة"],                     "ما هو nlp"),
    (["تعلم الآلة", "تعلم الالة", "تعلم آلي"],    "ما هو تعلم الآلة"),
    (["التعلم العميق", "تعلم عميق"],               "اشرح لي التعلم العميق"),
    (["الشبكة العصبية", "شبكة عصبية"],             "ما هي الشبكة العصبية"),
    (["المحوّل", "محول", "transformer"],            "ما هو المحوّل"),
    (["llm", "نموذج اللغة الكبير"],                "ما هو نموذج اللغة الكبير"),
    (["الذكاء الاصطناعي", "ذكاء اصطناعي", " ai "],  "ما هو الذكاء الاصطناعي"),
    (["تحليل المشاعر"],                            "ما هو تحليل المشاعر"),
    (["تصنيف النية"],                              "ما هو تصنيف النية"),
    (["الأرابيزي", "أرابيزي"],                     "ما هي الأرابيزي"),
    (["سيدار", "cedar"],                           "ما هو سيدار"),
    (["الترميز", "ترميز", "token"],                "ما هو الترميز"),
    (["روبوت الدردشة", "chatbot"],                 "ما هو روبوت الدردشة"),
    (["لبنان"],                                    "أخبرني عن لبنان"),
]

_KB_KEYWORD_MAP = [
    (["nlp", "natural language processing"],    "what is nlp"),
    (["machine learning", " ml "],              "what is machine learning"),
    (["deep learning"],                         "what is deep learning"),
    (["neural network"],                        "what is a neural network"),
    (["transformer"],                           "what is a transformer"),
    (["bert"],                                  "what is bert"),
    (["gpt"],                                   "what is gpt"),
    (["llm", "large language model"],           "what is llm"),
    (["artificial intelligence", " ai "],       "what is ai"),
    (["sentiment analysis"],                    "what is sentiment analysis"),
    (["intent classif"],                        "what is intent classification"),
    (["arabizi"],                               "what is arabizi"),
    (["cedar chatbot", "what is cedar"],        "what is cedar"),
    (["tokeniz"],                               "what is tokenization"),
    (["named entity", " ner "],                "what is named entity recognition"),
    (["blenderbot"],                            "what is blenderbot"),
    (["lebanon"],                               "tell me about lebanon"),
    (["chatbot"],                               "what is a chatbot"),
]

ARABIZI_KEYWORD_MAP = [
    (["nlp", "natural language", "lugha"],          "what is nlp"),
    (["machine learning", "ta3allum aleh"],         "what is machine learning"),
    (["deep learning", "ta3allum 3amiq"],           "what is deep learning"),
    (["neural", "3asabi", "shabake"],               "what is a neural network"),
    (["transformer"],                               "what is a transformer"),
    (["bert"],                                      "what is bert"),
    (["gpt"],                                       "what is gpt"),
    (["llm", "large language"],                     "what is llm"),
    (["ai", "zaka2", "zaka", "zeka"],               "what is ai"),
    (["sentiment", "masha3er"],                     "what is sentiment analysis"),
    (["intent", "niyye", "niyeh"],                  "what is intent classification"),
    (["arabizi", "3arabizi"],                       "what is arabizi"),
    (["cedar", "sidr"],                             "what is cedar"),
    (["token"],                                     "what is tokenization"),
    (["ner", "named entity"],                       "what is named entity recognition"),
    (["blenderbot"],                                "what is blenderbot"),
    (["lebnen", "lebnan", "lebanon", "loubnan"],    "tell me about lebanon"),
    (["chatbot", "bot", "robot"],                   "what is a chatbot"),
]

ARABIZI_CHITCHAT_PHRASES = {
    "shu 3am ta3mel":  ["Ana hon 3am sa3ed el nas! Shu fik bsa3dak?", "3am ne7ke ma3ak! Shu baddak ta3ref?"],
    "shu 3am ti3mel":  ["Ana hon 3am sa3ed el nas! Shu fik bsa3dak?", "3am ne7ke ma3ak! Shu baddak ta3ref?"],
    "shu 3am ta3mle":  ["Ana hon 3am sa3ed el nas! Shu fik bsa3dak?", "3am ne7ke ma3ak! Shu baddak ta3ref?"],
    "shu fik":         ["Ana mni7 hamdellah! W enta?", "Tamam! Shu baddak?"],
    "shu akhbarak":    ["Kello tamam hamdellah! W enta shu akhbarak?", "Mni7 ktir! Shu fi jdid ma3ak?"],
    "shu el akhbar":   ["Kello tamam! W enta shu akhbarak?", "Mni7 hamdellah! Shu fi jdid?"],
    "shu fi jdid":     ["Wallah kell shi tamam! W enta?", "Ma fi shi jdid ktir, w enta shu 3amel?"],
    "wenak":           ["Ana hon dayman! Shu baddak?", "Mawjoud! Shu fik bsa3dak?"],
    "waynak":          ["Ana hon dayman! Shu baddak?", "Mawjoud! Shu fik bsa3dak?"],
    "3afye":      ["Allah y3afik! 🌲", "Allah y3afik, yslamo!", "Yslamo! Allah y3afik."],
    "3afe":       ["Allah y3afik! 🌲", "Allah y3afik, yslamo!"],
    "yslamo":     ["Allah ysallmak! 🌲", "Tslam! Ahla fiik."],
    "yeslamo":    ["Allah ysallmak! 🌲", "Tslam! Ahla fiik."],
    "sahtein":    ["3a albak! 🌲", "Sahtein 3aleik!"],
    "mabrouk":    ["Allah ybarek fik! 🌲", "Yeslamo, Allah ybarek!"],
    "n3eeman":    ["Na3iman 3aleik! 🌲", "Allah yn3am 3aleik!"],
    "tslam":      ["Tslam ana kamen! 🌲", "Yslamo, Allah y5allik!"],
    "tamam":      ["Mni7! W enta, tamam?", "Hamdellah! Shu 3am bysir ma3ak?"],
    "mni7":       ["Ktir mni7! Shu fi jdid?", "Ktir mni7! Shu 3am ta3mel?"],
    "wallah":     ["Wallah sah! Shu baddak t3arraf kamen?", "Eh wallah, 2ellak shu..."],
    "machi":      ["Eh machi l7al! Shu akhbarak el ba2i?", "Machi w tamam, hamdellah!"],
    "hamdellah":  ["Hamdellah dayman! W enta, keefak?", "Ktir mni7, hamdellah!"],
    "yalla":      ["Yalla habibi, shu baddak na3mel?", "Yalla inshallah! Shu 3andak bi balik?"],
    "inshallah":  ["Inshallah dayman el kher! Shu baddak?", "Inshallah kella tamam!"],
    "shu baddak": ["Sa2al w ana hon la sa3dak habibi!", "2ellak shu baddak w bsa3dak!"],
    "keefak":     ["Ana mni7 hamdellah! W enta, keefak?", "Tamam tamam, shukran!"],
    "kifak":      ["Ana mni7 hamdellah! W enta?", "Tamam, shukran! W enta kifak?"],
}

ARABIZI_FALLBACK_RESPONSES = [
    "Soual 7elo! Bas ma 3ande ma3loumat kafi 3an hala2. Fi2ik tsa2al 3an AI aw NLP?",
    "Msh fahemha ktir, fi2ik t3id el soual? Ana kbir bi AI w machine learning!",
    "Wallah ma 3ande jawab 3an haydal mawdou3. Bas 2id2allne bi AI, deep learning, aw lebnen!",
]

ARABIC_FALLBACK_RESPONSES = [
    "هذا سؤال مثير للاهتمام! للأسف معلوماتي عن هذا الموضوع محدودة حالياً. هل تريد أن تسألني عن الذكاء الاصطناعي أو معالجة اللغة الطبيعية؟",
    "سؤال حلو! ما عندي جواب كافي هلأ على هذا الموضوع، بس بقدر ساعدك بأسئلة عن الذكاء الاصطناعي والتكنولوجيا.",
    "ما قدرت أفهم قصدك بشكل كامل. فيك تعيد السؤال بطريقة مختلفة؟ أو اسألني عن الذكاء الاصطناعي، تعلم الآلة، أو لبنان.",
    "والله ما عندي معلومات كافية عن هذا الموضوع. بس أنا خبير في الذكاء الاصطناعي ومعالجة اللغة — اسألني!",
]

ARABIC_CHITCHAT_RESPONSES = [
    "أنا منيح الحمد لله! وأنت، شو أخبارك؟",
    "تمام تمام، الله يسلمك! شو عم تعمل؟",
    "الحمد لله منيح! وأنت كيفك؟",
    "والله منيح، شكراً! شو في جديد؟",
]

ARABIZI_GENERIC_CHITCHAT = [
    "Ana mni7 hamdellah! W enta, shu akhbarak?",
    "Tamam tamam, allah yesalmak! Shu 3am ta3mel?",
    "Hamdellah mni7! W enta, keefak?",
    "Wallah mni7, shukran! Shu fi jdid?",
]

ARABIZI_RESPONSES = {
    Intent.GREETING: [
        "Ahla w sahla! Kifak/kifik? 🌲",
        "Marhaba! Shu akhbarak lyom?",
        "Hala wallah! Keefak ya zalameh?",
        "Hey! Ahla fiik, shu 3am ta3mel?",
    ],
    Intent.FAREWELL: [
        "Yalla bye! Allah ma3ak 👋",
        "Ma3 el salame! Nshallah mne7ke ba3den",
        "Bye habibi! Take care ya zalameh",
    ],
    Intent.QUESTION: [
        "Soual mni7! {answer}",
        "Ah, soual 7elo! {answer}",
        "Hala2 bi jawbak: {answer}",
    ],
    Intent.THANKS: [
        "3afwan habibi! Ay wa2et 🌲",
        "Tekram ya zalameh! Ana hon dayman",
        "La shukr 3a wajeb!",
    ],
    Intent.COMPLAINT: [
        "Sorry ya zalameh, khalline 7awel sa3dak a7san",
        "Ma3lesh, 7a2ak 3layye. Shu fi2e sa3dak?",
    ],
    Intent.FEEDBACK: [
        "Merci 3al feedback! Bi sa3edne ktir",
        "Shukran! Ra7 7awel et7assan",
    ],
    Intent.CHITCHAT: [
        "Wallah! {answer}",
        "Eh sah, {answer}",
        "Ahh mni7 — {answer}",
    ],
    Intent.REQUEST: [
        "Akid habibi! {answer}",
        "Tab3an: {answer}",
        "Inshallah bsa3dak! {answer}",
    ],
    Intent.UNKNOWN: [
        "Hmm, ma fhemet ktir. Fi2ik t3id el soual?",
        "Sorry, ma 2dert efham. Shu 2asdak?",
    ],
}

ARABIC_RESPONSES = {
    Intent.GREETING: [
        "أهلاً وسهلاً! كيفك اليوم؟ 🌲",
        "مرحبا! شو أخبارك؟",
        "هلا والله! أهلاً فيك",
    ],
    Intent.FAREWELL: [
        "مع السلامة! الله معك 👋",
        "باي! إنشاالله منحكي بكرا",
        "يلا باي! تصبح على خير",
    ],
    Intent.QUESTION: [
        "سؤال منيح! {answer}",
        "هلأ بجاوبك: {answer}",
        "يعني، {answer}",
    ],
    Intent.THANKS: [
        "عفواً حبيبي! أي وقت 🌲",
        "تكرم! أنا هون دايماً",
        "لا شكر على واجب!",
    ],
    Intent.COMPLAINT: [
        "معلش، حقك عليي. شو فيني ساعدك؟",
        "آسف! خليني حاول أحسن المرة الجاية",
    ],
    Intent.FEEDBACK: [
        "شكراً عالملاحظات! بتساعدني كتير",
        "ميرسي! رح حاول إتحسن",
    ],
    Intent.CHITCHAT: [
        "والله! {answer}",
        "إيه صح، {answer}",
        "آه منيح! {answer}",
    ],
    Intent.REQUEST: [
        "أكيد حبيبي! {answer}",
        "طبعاً! {answer}",
        "إنشاالله بساعدك! {answer}",
    ],
    Intent.UNKNOWN: [
        "ما فهمت كتير، فيك تعيد السؤال؟",
        "معلش ما قدرت إفهم، شو قصدك؟",
    ],
}


ENGLISH_RESPONSES = {
    Intent.GREETING: [
        "Hello! I'm Cedar 🌲 — how can I help you today?",
        "Hi there! Ask me anything about AI, NLP, or languages.",
        "Hey! I'm Cedar, your trilingual assistant. What's on your mind?",
        "Welcome! I can chat in English, Arabic, and Lebanese Arabizi. How can I help?",
    ],
    Intent.FAREWELL: [
        "Goodbye! Come back anytime 🌲",
        "Take care! Feel free to return with more questions.",
        "See you later! 👋",
        "Bye for now — it was nice talking with you!",
    ],
    Intent.THANKS: [
        "You're welcome! Happy to help 🌲",
        "Anytime! Let me know if you need anything else.",
        "My pleasure! What else can I do for you?",
        "Glad I could help!",
    ],
    Intent.COMPLAINT: [
        "I'm sorry about that — let me try to help better.",
        "Apologies! Could you tell me more so I can assist?",
        "You're right, let me try again.",
    ],
    Intent.FEEDBACK: [
        "Thanks for the feedback — it helps me improve!",
        "Appreciate it! I'll keep working to do better.",
        "Thank you for letting me know!",
    ],
    Intent.CHITCHAT: [
        "I'm doing well, thanks for asking! How about you?",
        "All good here 🌲 — what would you like to talk about?",
        "I'm just here ready to help! What's up?",
    ],
}


class CedarChatbot:
    DEFAULT_MODEL = "facebook/blenderbot-400M-distill"

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_turns: int = 10,
        max_length: int = 128,
        device: str = "cpu",
        use_multilingual_decoder: bool = True,
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.device = device
        self.use_multilingual_decoder = use_multilingual_decoder

        self.normalizer = ArabiziNormalizer()
        self.lang_detector = LanguageDetector()
        self.intent_classifier = IntentClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.memory = ConversationMemory(max_turns=max_turns)

        self._model = None
        self._tokenizer = None
        self._load_model()

        self.multilingual = None
        if self.use_multilingual_decoder:
            try:
                self.multilingual = MultilingualGenerator(device=device)
            except Exception as e:
                logger.warning(f"Multilingual decoder unavailable, falling back to templates: {e}")
                self.multilingual = None

    def _load_model(self):
        try:
            logger.info(f"Loading model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _is_identity_question(self, message: str, language: Language) -> bool:
        cleaned = message.lower().strip().rstrip("?!.,؟،")
        if language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC):
            return any(p in cleaned for p in IDENTITY_PATTERNS_AR)
        if language == Language.LEBANESE_ARABIZI:
            if any(p in cleaned for p in IDENTITY_PATTERNS_ARABIZI):
                return True
            return any(p in cleaned for p in IDENTITY_PATTERNS_EN)
        return any(p in cleaned for p in IDENTITY_PATTERNS_EN)

    def _identity_response(self, language: Language) -> str:
        if language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC):
            return IDENTITY_RESPONSE_AR
        if language == Language.LEBANESE_ARABIZI:
            return IDENTITY_RESPONSE_ARABIZI
        return IDENTITY_RESPONSE_EN

    def _knowledge_lookup(self, message: str) -> Optional[str]:
        cleaned = message.lower().strip().rstrip("?!.,")

        if cleaned in KNOWLEDGE_BASE:
            return KNOWLEDGE_BASE[cleaned]

        for prefix in ("explain ", "describe ", "define ", "tell me about ",
                        "what do you know about ", "can you explain "):
            if cleaned.startswith(prefix):
                remainder = cleaned[len(prefix):]
                key = f"what is {remainder}"
                if key in KNOWLEDGE_BASE:
                    return KNOWLEDGE_BASE[key]

        padded = f" {cleaned} "
        for keywords, kb_key in _KB_KEYWORD_MAP:
            if any(kw in padded for kw in keywords):
                answer = KNOWLEDGE_BASE.get(kb_key)
                if answer:
                    return answer

        return None

    def _arabic_knowledge_lookup(self, message: str) -> Optional[str]:
        cleaned = message.strip().rstrip("؟?!.,")

        if cleaned in ARABIC_KNOWLEDGE_BASE:
            return ARABIC_KNOWLEDGE_BASE[cleaned]

        for prefix in (
            "اشرح لي ", "اشرح ", "عرّف ", "عرف ", "ما هو ", "ما هي ",
            "ما معنى ", "أخبرني عن ", "حدثني عن ", "ما هي تفاصيل ",
        ):
            if cleaned.startswith(prefix):
                remainder = cleaned[len(prefix):]
                for key in (f"ما هو {remainder}", f"ما هي {remainder}", remainder):
                    if key in ARABIC_KNOWLEDGE_BASE:
                        return ARABIC_KNOWLEDGE_BASE[key]

        for keywords, kb_key in _AR_KB_KEYWORD_MAP:
            if any(kw in cleaned for kw in keywords):
                answer = ARABIC_KNOWLEDGE_BASE.get(kb_key)
                if answer:
                    return answer

        return None

    def _arabizi_knowledge_lookup(self, message: str) -> Optional[str]:
        cleaned = message.lower().strip().rstrip("?!.,")
        padded = f" {cleaned} "

        for keywords, kb_key in ARABIZI_KEYWORD_MAP:
            if any(kw in padded for kw in keywords):
                answer = ARABIZI_KNOWLEDGE_BASE.get(kb_key)
                if answer:
                    return answer
                answer = KNOWLEDGE_BASE.get(kb_key)
                if answer:
                    return answer

        return self._knowledge_lookup(message)

    def _arabizi_chitchat_lookup(self, message: str) -> Optional[str]:
        cleaned = message.lower().strip().rstrip("?!.,")
        for keyword, responses in ARABIZI_CHITCHAT_PHRASES.items():
            if keyword in cleaned:
                return random.choice(responses)
        return None

    @staticmethod
    def _is_echo(generated: str, *sources: str) -> bool:
        g = generated.lower().strip().rstrip("?!.,؟،")
        if len(g) < 9:
            return True
        for src in sources:
            if not src:
                continue
            s = src.lower().strip().rstrip("?!.,؟،")
            if g == s or g in s or s in g:
                return True
        return False

    @staticmethod
    def _looks_like_social(message: str) -> bool:
        cleaned = message.lower().strip().rstrip("?!.,؟،")
        words = cleaned.split()
        if len(words) <= 4:
            return True
        topic_markers = (
            "shu", "what", "explain", "ishra7", "esra7", "2elli", "elli",
            "7akine", "7kini", "3an", "about", "ma hu", "ما هو", "ما هي",
            "اشرح", "حدثني", "عن",
        )
        return not any(m in cleaned for m in topic_markers)

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ChatResponse:
        start_time = time.time()

        session = self.memory.get_or_create_session(session_id, user_id)
        sid = session.session_id

        lang_result = self.lang_detector.detect(message)

        normalized = message
        if lang_result.language == Language.LEBANESE_ARABIZI:
            normalized = self.normalizer.normalize(message)

        intent_result = self.intent_classifier.classify(message)
        sentiment_result = self.sentiment_analyzer.analyze(message)

        user_msg = self.memory.add_message(
            sid, "user", normalized,
            metadata={
                "original": message,
                "language": lang_result.language.value,
                "intent": intent_result.intent.value,
                "sentiment": sentiment_result.label.value,
            }
        )

        is_arabic = lang_result.language in (Language.ARABIC_MSA, Language.LEBANESE_ARABIC)
        is_arabizi = lang_result.language == Language.LEBANESE_ARABIZI
        knowledge_hit = False
        generator_used = "templates"

        if self._is_identity_question(message, lang_result.language) or \
                (is_arabizi and self._is_identity_question(normalized, Language.LEBANESE_ARABIZI)):
            response_text = self._identity_response(lang_result.language)
            knowledge_hit = True
            generator_used = "identity_intercept"

        elif is_arabic:
            response_text, generator_used = self._handle_arabic(message, intent_result)
            knowledge_hit = intent_result.intent not in (
                Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                Intent.COMPLAINT, Intent.FEEDBACK, Intent.CHITCHAT,
            )

        elif is_arabizi:
            response_text, generator_used = self._handle_arabizi(message, normalized, intent_result)
            knowledge_hit = intent_result.intent not in (
                Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                Intent.COMPLAINT, Intent.FEEDBACK, Intent.CHITCHAT,
            )

        else:
            social_intents = (
                Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                Intent.COMPLAINT, Intent.FEEDBACK,
            )
            english_answer = self._knowledge_lookup(message)
            if english_answer:
                response_text = english_answer
                knowledge_hit = True
                generator_used = "knowledge_base"
            elif intent_result.intent in social_intents:
                response_text = random.choice(ENGLISH_RESPONSES[intent_result.intent])
                generator_used = "templates"
            else:
                context = self.memory.get_context(sid)
                response_text = self._generate(context, message)
                generator_used = "blenderbot"

        self.memory.add_message(sid, "assistant", response_text)

        elapsed = round((time.time() - start_time) * 1000, 1)

        return ChatResponse(
            response=response_text,
            session_id=sid,
            message_id=user_msg.message_id,
            metadata={
                "detected_language": lang_result.language.value,
                "language_confidence": lang_result.confidence,
                "normalized_text": normalized if normalized != message else None,
                "intent": intent_result.intent.value,
                "intent_confidence": intent_result.confidence,
                "sentiment": {
                    "label": sentiment_result.label.value,
                    "score": sentiment_result.score,
                },
                "model": self.model_name,
                "generator": generator_used,
                "response_time_ms": elapsed,
                "knowledge_hit": knowledge_hit,
            },
        )

    def _handle_arabic(self, message: str, intent_result) -> tuple:
        intent = intent_result.intent

        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                      Intent.COMPLAINT, Intent.FEEDBACK) and self._looks_like_social(message):
            return self._to_arabic_response("", intent), "templates"

        arabic_fact = self._arabic_knowledge_lookup(message)
        english_fact = self._knowledge_lookup(message) if not arabic_fact else None
        grounding = arabic_fact or english_fact or ""

        if self.multilingual:
            generated = self.multilingual.generate_arabic(message, context_fact=grounding)
            if generated and not self._is_echo(generated, message):
                return generated, "mt5_decoder"

        if arabic_fact:
            return self._to_arabic_response(arabic_fact, Intent.QUESTION), "templates+kb"
        if english_fact:
            return self._to_arabic_response(english_fact, Intent.QUESTION), "templates+kb"
        if intent == Intent.CHITCHAT:
            return random.choice(ARABIC_CHITCHAT_RESPONSES), "templates"

        return random.choice(ARABIC_FALLBACK_RESPONSES), "templates"

    def _handle_arabizi(self, message: str, normalized: str, intent_result) -> tuple:
        intent = intent_result.intent

        chitchat_response = self._arabizi_chitchat_lookup(message)
        if chitchat_response:
            return chitchat_response, "templates"

        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                      Intent.COMPLAINT, Intent.FEEDBACK) and self._looks_like_social(message):
            return self._to_arabizi_response("", intent), "templates"

        arabizi_fact = self._arabizi_knowledge_lookup(message)
        if arabizi_fact:
            return self._to_arabizi_response(arabizi_fact, Intent.QUESTION), "templates+kb"

        arabic_fact = self._arabic_knowledge_lookup(normalized)
        if arabic_fact:
            return self._to_arabizi_response(arabic_fact, Intent.QUESTION), "templates+kb"

        if self.multilingual:
            generated = self.multilingual.generate_arabizi(message, normalized=normalized, context_fact="")
            if generated and not self._is_echo(generated, message, normalized):
                return generated, "mt5_decoder"

        if intent == Intent.CHITCHAT:
            return random.choice(ARABIZI_GENERIC_CHITCHAT), "templates"

        return random.choice(ARABIZI_FALLBACK_RESPONSES), "templates"

    def _to_arabic_response(self, raw_response: str, intent: Intent) -> str:
        templates = ARABIC_RESPONSES.get(intent, ARABIC_RESPONSES[Intent.CHITCHAT])
        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                      Intent.COMPLAINT, Intent.FEEDBACK):
            return random.choice(templates)
        template = random.choice(templates)
        if "{answer}" in template:
            if raw_response:
                return template.format(answer=raw_response)
            return random.choice(ARABIC_RESPONSES.get(Intent.UNKNOWN, ["ما فهمت، فيك تعيد؟"]))
        return template

    def _to_arabizi_response(self, raw_response: str, intent: Intent) -> str:
        templates = ARABIZI_RESPONSES.get(intent, ARABIZI_RESPONSES[Intent.CHITCHAT])
        if intent in (Intent.GREETING, Intent.FAREWELL, Intent.THANKS,
                      Intent.COMPLAINT, Intent.FEEDBACK):
            return random.choice(templates)
        template = random.choice(templates)
        if "{answer}" in template:
            if raw_response:
                return template.format(answer=raw_response)
            return random.choice(ARABIZI_FALLBACK_RESPONSES)
        return template

    def _generate(self, context: str, message: str) -> str:
        try:
            input_text = f"{context} \n {message}" if context else message
            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return "I'm sorry, I encountered an error. Could you try again?"

    def get_history(self, session_id: str) -> list:
        return self.memory.get_history(session_id)

    def submit_feedback(self, session_id: str, message_id: str, rating: int, comment: str = ""):
        self.memory.add_feedback(session_id, message_id, rating, comment)

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "multilingual_decoder": "mt5-small" if self.multilingual else "disabled",
            **self.memory.get_stats(),
        }