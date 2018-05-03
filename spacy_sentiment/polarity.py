import spacy
from spacy.tokens import Doc
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def polarity_scores(doc):
    return sentiment_analyzer.polarity_scores(doc.text)


Doc.set_extension('polarity_scores', getter=polarity_scores)
nlp = spacy.load('en')
doc = nlp("Ok - so I don't love Pete and I went here (again!). It's been 9 years since my last meal here. Why you ask? I was in the neighborhood and real hungry; just wanted something simple and fast. Plus McD's was closed (for a long while now).This place is the saddest restaurant I have ever seen/been to. Sadder than I remember last time. Very hard to describe -- felt almost like I was in an episode of the Twilight Zone. There is no smell of food. Half the patrons (there about 8 people) are sitting there for no particular reason - no one was eating. There are exactly 3 workers: 1 in the back cooking (ha ha - there wasn't any real cooking going on here!), 1 miserable cashier (he really looked it!), 1 manager.The food magically appears from the lady out back, the guy adds the french fries, and makes my ice coffee. Oh the coffee! He chucks in the ice and gets a jug the mini fridge and pours it in my cup. The coffee jug had something like a liquor spout. Junior whopper was cold (but not smashed this time), fries were hot, ice coffee was pretty thick. I got food in the my belly and thanking my lucky stars that my health was in tact.")
print(doc._.polarity_scores)

