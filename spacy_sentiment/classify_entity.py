import spacy
nlp = spacy.load('en')
doc = nlp(u'Bk equals Burger King and Mcpyth')

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)