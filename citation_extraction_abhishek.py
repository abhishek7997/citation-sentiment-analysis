# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 19:23:21 2022

@author: Abhi
"""

from io import StringIO
import re
import unidecode
import csv
import string

from pdfminer.high_level import extract_text
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

import nltk
import nltk.tag
import nltk.data
from nltk.sentiment import SentimentIntensityAnalyzer

# =======================================
# ============== CONSTANTS ==============
# =======================================
PAPERS = {
    'text1': "./abhishek/Document-level sentiment classification An empirical comparison between SVM and ANN.pdf",
    'text2': "./abhishek/Lexicon-Based Methods for Sentiment Analysis.pdf",
    'text3': "./abhishek/Sentiment analysis A combined approach.pdf",
    'text4': "./abhishek/Sentiment Analysis and Opinion Mining A survey.pdf",
    'text5': "./abhishek/Sentiment Strength Detection in Short Informal Text.pdf",
    'text6': "./abhishek/Scopus as data source.pdf",
    # 'text7': "./abhishek/Learning sentiment specific word.pdf"
}

OUTPUTFILE = "results_abhishek.txt"
POSITIVE_WORDS_FILE = "positive-words.txt"
NEGATIVE_WORDS_FILE = "negative-words.txt"

# NLTK_WORDS = set(nltk.corpus.words.words())
# FOOTER_TEXTS = ["DOI:", "doi:", "Ltd", "All rights reserved"]

CITATIONS_REGEX_V1 = r"[\w]+[ ][&][ ][\w]+[,][ ][\d]+"
# CITATIONS_REGEX_v2 = r"\b(?!(?:Although|Also)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:, *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))"
CITATIONS_REGEX_V3 = r"\b(?!(?:Although|Also)\b)(?:[A-Za-z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:,? *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))"
CITATIONS_REGEX_V4 = r"^(?:[A-Z](?:(?!$)[A-Za-z\s&.,'’])+)\((?:\d{4})\)\.?\s*(?:[^()]+?[?.!])\s*(?:(?:(?:(?:(?!^[A-Z])[^.]+?)),\s*(?:\d+)[^,.]*(?=,\s*\d+|.\s*Ret))|(?:In\s*(?:[^()]+))\(Eds?\.\),\s*(?:[^().]+)|(?:[^():]+:[^().]+\.)|(?:Retrieved|Paper presented))"
CITATIONS_REGEX_V5 = r"\b(?!(?:Although|Also)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:(?:,? |,*)*(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))"

CITATIONS_REGEX = re.compile("(%s|%s|%s|%s)" % (
    CITATIONS_REGEX_V1, CITATIONS_REGEX_V3, CITATIONS_REGEX_V4, CITATIONS_REGEX_V5))

REFERENCES_REGEX_V1 = r"^(?:[A-Za-z](?:(?:(?!$)[A-Za-z\s&.,'’-]+)\(?[0-9]{4}(?:,? ?[A-Za-z]*)?\)?))."
REFERENCES_REGEX_V2 = r"^(?:[A-Za-z](?:(?:(?!$)[A-Za-z\s&.,'’-]+)\(?[0-9]{4}(?:,? ?(?:January|February|March|April|May|June|July|August|September|October|November|December))?\)?))."
references_regex = re.compile("(%s|%s)" % (
    REFERENCES_REGEX_V1, REFERENCES_REGEX_V2), re.MULTILINE)

YEARS_REGEX = r"[\d]{4}"

stopwords = nltk.corpus.stopwords.words("english")
stopwords.remove('not')


def ReadFile(fileName):
    '''Read the file and return list of words'''
    with open(fileName, 'r') as file:
        return [word.strip() for word in file.readlines()]

# ACCENTED CHARACTERS TO ASCII


def unicodeToASCII(unicode_text):
    '''Replace all non-english characters from extracted text'''
    # new_text = unicode_text.replace("\\p{No}+", "")
    new_text = re.sub(r"[¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞↉]+", '', unicode_text)
    converted_text = unidecode.unidecode(new_text).strip()
    return converted_text


def removeUnicode(text):
    '''Remove all non-english characters from extracted text'''
    return ''.join([i if ord(i.lower()) < 128 else ' ' for i in text])


def readPDF(fileName=None):
    '''Extract text from PDF File'''
    if not isinstance(fileName, str):
        raise TypeError("String must be provided")
    if not fileName:
        raise Exception("Filename not provided!")
    text = extract_text(fileName)
    text = text.strip().lower()
    text = unicodeToASCII(text)
    text = removeUnicode(text)
    return text.strip()


def readPDFBuffered(fileName=None):
    '''Extract text from PDF File'''
    if not isinstance(fileName, str):
        raise TypeError("String must be provided")
    if not fileName:
        raise Exception("Filename not provided!")

    output_string = StringIO()
    with open(fileName, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()


def readPageTextIncremented(fileName=None):
    '''Extract text from PDF File page by page'''
    if not isinstance(fileName, str):
        raise TypeError("String must be provided")
    if not fileName:
        raise Exception("Filename not provided!")

    texts = []
    for page_layout in extract_pages(fileName):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                content = element.get_text().strip()
                content = unicodeToASCII(content)
                content = removeUnicode(content)
                texts.append(content)

    return texts


def restructureText(pageTexts):
    '''Properly format the raw text'''
    newText = ''

    for line in pageTexts:
        lines = line.strip().split('\n')
        newLine = ' '.join(list(lines)) + '\n'
        newText += newLine + '\n'

    # newText = newText.replace('- ', '')
    newText = newText.replace('" ', '')
    newText = re.sub(' +', ' ', newText)
    newText = re.sub(r'-\s+', '', newText)
    return newText


def parenthesizeCitationsYears(text):
    '''Place open and close brackets around year'''
    years = set(re.findall(YEARS_REGEX, text))

    for year in years:
        text = text.replace(year, '(' + year + ')')

    return text


def getCitations(text):
    '''Extract all citations from text (Regex)'''
    return sorted(list(set(re.findall(CITATIONS_REGEX, text))))


def getCitationsNLTK(text):
    '''Extract all citations from text (NLTK)'''
    return sorted(list(set(nltk.regexp_tokenize(text, CITATIONS_REGEX))))


def getCitationsLocations(text):
    '''Extract the starting and ending indices of all citations'''
    citation_locations = []
    for st in CITATIONS_REGEX.finditer(text):
        citation_locations.append((st.group(), st.span()))

    return citation_locations


def getParagraphs(text):
    '''Return list of paragraphs from raw text'''
    return text.split('\n\n')


def getReferences(text):
    '''Return just the text which contains all references'''
    pos = text.rfind("References")
    return text[pos::]


def getReferencesList(references_text):
    '''Returns list of all references from the raw text containing all references'''
    return sorted(list(set(re.findall(references_regex, references_text))))


def tokenizeText(text):
    '''Tokenize the raw text using NLTK'''
    tokens = nltk.word_tokenize(text)
    return tokens


def getNLTKText(tokens):
    '''Returns text that can be operated upon by NLTK'''
    final_tokens = [token.lower() for token in tokens if (
        token.isalpha() and (token.lower() not in stopwords))]
    # cleaned_tokens = [tok for tok in final_tokens if tok.lower() not in stopwords]
    parsed_text = nltk.Text(final_tokens)
    return parsed_text


def POSTagger(parsed_text):
    '''Apply Part-Of-Speech tagger on words'''
    return nltk.pos_tag(parsed_text)


def getAdjectives(tagged_results):
    '''Return list of adjectives from Part-Of-Speech tokens'''
    adjectives = [(re.sub(r'[^A-Za-z]', '', word), tag)
                  for (word, tag) in tagged_results if tag in ('JJ', 'JJR', 'JJS') and len(re.sub(r'[^A-Za-z]', '', word)) > 2]
    return adjectives


def getAdjectivesWords(tagged_results):
    '''Return list of adjectives from Part-Of-Speech tokens (words only)'''
    return list({
        re.sub(r'[^A-Za-z]', '', word) for (word, token) in tagged_results if (token in ('JJ', 'JJS', 'JJR') and len(re.sub(r'[^A-Za-z]', '', word)) > 2)})


def getPositiveAdjectives(adjectives):
    '''Return list of adjectives that are in the list of positive words'''
    positive_adjectives = [
        adjective for adjective in adjectives if adjective in POSITIVE_WORDS]
    return len(positive_adjectives), positive_adjectives


def getNegativeAdjectives(adjectives):
    '''Return list of adjectives that are in the list of negative words'''
    negative_adjectives = [
        adjective for adjective in adjectives if adjective in NEGATIVE_WORDS]
    return len(negative_adjectives), negative_adjectives


def getCitationsParagraphLocation(text, citations_list):
    '''Return list of citations and their enclosing paragraph'''
    result = []
    reference_pos = text.rfind("References")
    search_text = text[0:reference_pos]
    citation_paragraphs = getParagraphs(search_text)
    for citation in citations_list:
        matches = ' '.join(
            [re.sub(' +', ' ', para) for para in citation_paragraphs if citation in para and len(para) > 10]).strip()
        matches = matches.replace('- ', '')
        if len(matches) > 10:
            result.append({citation: matches})
    return result


def citeText(text, citations_list):
    '''Replace all citation text with a special token string <CIT${citation_index}$>'''
    for idx, citation in enumerate(citations_list):
        text = text.replace(citation, f"<CIT${idx}$>")
    return text


def getSentiments(dictionary):
    '''Return the sentiment of citations from their paragraphs using inbuilt nltk Sentiment Intensity Analyzer'''
    results = []
    sia = SentimentIntensityAnalyzer()
    for item in dictionary:
        for (citation, paragraph) in item.items():
            if len(paragraph) > 10:
                sentiment = sia.polarity_scores(paragraph)
                results.append((citation, paragraph, sentiment))
    return results


def getCitationsData(dictionary):
    '''Return list of citations, its pargraph and adjectives'''
    results = []
    # adjectives = getAdjectives(blob.tags)
    for item in dictionary:
        for (citation, paragraph) in item.items():
            if len(paragraph) > 10:
                # parsed_text = getNLTKText(paragraph)
                tagged_result = POSTagger(paragraph.lower().split())
                adjectives = getAdjectivesWords(tagged_result)
                positive_adjectives = getPositiveAdjectives(adjectives)
                negative_adjectives = getNegativeAdjectives(adjectives)
                results.append((citation, paragraph, adjectives,
                               positive_adjectives, negative_adjectives))
    return results


def printSentiments(sentiments):
    '''Print the citation, its paragraph and its sentiment values'''
    print("\tCitation\t\t\t\tParagraph\t\t\t\tNeg\t  Neu\t  Pos\t\t Sentiment")
    for (citation, paragraph, scores) in sentiments:
        if len(paragraphs) > 10:
            sentiment = "Neutral"

            if scores['compound'] >= 0.5:
                sentiment = "Positive"
            elif scores['compound'] <= -0.5:
                sentiment = "Negative"

            print(
                f"{citation[:15]+'...'}\t\t|{paragraph[:15]+'...'}\t\t| {scores['neg']:.2f} || {scores['neu']:.2f} || {scores['pos']:.2f}\t\t{sentiment}")


def writeSentimentsToFile(sentiments):
    '''Print the citation, its paragraph and its sentiment values'''
    with open("default_sentiments.txt", "a") as file:
        for (citation, paragraph, scores) in sentiments:
            if len(paragraphs) > 10:
                sentiment = "Neutral"

                if scores['compound'] >= 0.5:
                    sentiment = "Positive"
                elif scores['compound'] <= -0.5:
                    sentiment = "Negative"

                file.write("=================================\n\n")
                file.write(f"Citation: {citation}\n\n")
                file.write(f"Paragraph: {paragraph}\n\n")
                file.write(f"Negative: {scores['neg']:.3f}\n\n")
                file.write(f"Neutral: {scores['neu']:.3f}\n\n")
                file.write(f"Positive: {scores['pos']:.3f}\n\n")
                file.write(f"Overall sentiment: {sentiment}\n\n")
                file.write("=================================\n\n")


def writeCitationsData(sentiments, file_name):
    '''Write all results to file'''
    # print("\tCitation\t\t\t\tParagraph\t\t\tPolarity\t\tSentiment")

    with open(f"{OUTPUTFILE}", "a") as file:
        file.write("=================================\n")
        file.write(f"Research Paper: {file_name}\n")
        file.write("=================================\n")

        pos_count = 0
        neg_count = 0
        neu_count = 0

        for (citation, paragraph, adjectives, (pos_adj, positive_adjectives), (neg_adj, negative_adjectives)) in sentiments:
           # print(f"{citation[:15]+'...'}\t\t| {paragraph[:15]+'...'}\t\t| {scores.polarity:.3f} | \t\t {sentiment}")

            file.write("Citation: " + citation + "\n\n")
            file.write("Paragraph: " + paragraph + "\n\n")
            # file.write("Scores: " + f"{scores.polarity:.4f}" + "\n\n")
            file.write(f"Adjectives : {adjectives}" + '\n\n')
            file.write(f"Positive Adjectives : {positive_adjectives}" + '\n\n')
            file.write(f"Negative Adjectives : {negative_adjectives}" + '\n\n')

            neu_count = len(adjectives) - (pos_adj + neg_adj)
            pos_count += pos_adj
            neg_count += neg_adj

            # file.write("=================================\n")

        file.write("Positive Adjectives : " + str(pos_count) + '\n')
        file.write("Negative Adjectives : " + str(neg_count) + '\n')
        file.write("Neutral Adjectievs : " + str(neu_count) + '\n')
        file.write("===========================================\n\n")


def writeCitationsToCSV(citations):
    '''Write citation data to CSV file'''
    with open("citations-data.csv", "w", newline="") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['citation', 'citation_text',
                         'adj', 'pos_adj', 'neg_adj'])
        for paper in citations:
            for (citation, citation_text, adjs, pos_adj, neg_adj) in paper:
                #csv_out.writerow((citation, citation_text))
                csv_out.writerow(
                    (citation, citation_text, adjs, pos_adj, neg_adj))


def getPolarityOfAdjectives(adjectives):
    '''Return list of positive, negative and neutral adjectives based on their sentiment calculated by Sentiment Intensity Analyzer'''
    sia = SentimentIntensityAnalyzer()
    positive_words = []
    negative_words = []
    neutral_words = []

    for (adjective, _) in adjectives:
        if (sia.polarity_scores(adjective)['compound']) >= 0.5:
            positive_words.append(adjective)
        elif (sia.polarity_scores(adjective)['compound']) <= -0.5:
            negative_words.append(adjective)
        else:
            neutral_words.append(adjective)

    return (list(set(negative_words)), list(set(positive_words)), list(set(neutral_words)))


def createUnigramTagger(citations_list):
    default_tagger = nltk.data.load(nltk.tag._POS_TAGGER)
    model = dict()
    for cit in citations_list:
        model[cit] = 'NNP'
    tagger = nltk.tag.UnigramTagger(model=model, backoff=default_tagger)
    return tagger


def citationsTagger(tagger, paragraphs):
    return tagger.tag(paragraphs)


def getAdjectivesFromCitationsData(citations_data):
    result = set()
    for (_, _, adjs, _, _) in citations_data:
        result.update(adjs)

    return sorted(list(result))


def writeAdjectivesToCSV(adjectives_list):
    '''Save list of adjectives to a CSV file'''
    rows = zip(adjectives_list, [0]*len(ALL_ADJECTIVES))
    with open('adjectives.csv', 'a', newline='') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("adjective", "score"))
        for row in rows:
            wr.writerow(row)


def removePunctuation(st):
    return st.strip().lower().translate(str.maketrans('', '', string.punctuation))


def getAllAuthors(citations):
    res = []
    for (cit, _, _, _, _) in citations:
        temp = removePunctuation(cit).split()
        res.append(temp[0])

    return res


def hIndex(citations, author):
    lst = []

    for idx, paper in enumerate(citations):
        res = getAllAuthors(paper)
        lst.append({'index': idx+1, 'citations': res.count(author)})

    sortedList = sorted(lst, key=lambda x: x['citations'], reverse=True)
    # return sortedList

    for i, cited in enumerate(sortedList):
        # finding current result
        result = len(sortedList) - i

        # if result is less than or equal
        # to cited then return result
        if result <= cited['citations']:
            return result
    return 0


POSITIVE_WORDS = ReadFile(POSITIVE_WORDS_FILE)
NEGATIVE_WORDS = ReadFile(NEGATIVE_WORDS_FILE)

ALL_CITATIONS = []
ALL_ADJECTIVES = []

open(f"{OUTPUTFILE}", 'w').close()
open("default_sentiments.txt", 'w').close()

for (_, currentText) in PAPERS.items():
    # pages = readPageTextIncremented(currentText)
    pages = readPDF(currentText).split('\n\n')
    restructuredText = restructureText(pages)
    # parenthesizedText = parenthesizeCitationsYears(restructuredText)
    citations = getCitations(restructuredText)
    locations = getCitationsLocations(restructuredText)
    paragraphs = getParagraphs(restructuredText)
    references = getReferences(restructuredText)
    references_list = getReferencesList(references)
    cited_text = citeText(restructuredText, citations)

    citations_dictionary = getCitationsParagraphLocation(
        restructuredText, citations)

    # ============== TOKENIZE ===============
    # tokens = tokenizeText(restructuredText)
    # parsed_text = getNLTKText(tokens)

    # ============== BIGRAMS ================
    # bigrams = list(nltk.bigrams(tokens))

    # ============== TRIGRAMS ================
    # trigrams = list(nltk.trigrams(final_tokens))

    # ========== APPLY P-O-S Tagger =========
    # tagged_result = POSTagger(parsed_text)

    # ========== EXTRACT ADJECTIVES =========
    # adjectives = set(getAdjectives(tagged_result))
    # set_adjectives = sorted(set(adjectives))

    # ========== CREATE FREQUENCY ===========
    # ========== DISTRIBUTION OF ============
    # ========== ADJECTIVES =================
    # tag_fd = nltk.FreqDist(adjectives)

    # ========== CITATIONS DATA ===========
    citations_data = getCitationsData(citations_dictionary)
    ALL_CITATIONS.append(citations_data)

    # ========== ADJECTIVES DATA ===========
    ALL_ADJECTIVES.extend(getAdjectivesFromCitationsData(citations_data))
    ALL_ADJECTIVES = sorted(list(set(ALL_ADJECTIVES)))

    writeAdjectivesToCSV(ALL_ADJECTIVES)

    # print("============ SENTIMENTS (NLTK SIA) =============")
    sentiments = getSentiments(citations_dictionary)
    # printSentiments(sentiments)
    writeSentimentsToFile(sentiments)

    # print("============ WRITE TO FILE =============")
    writeCitationsData(citations_data, currentText)
# neg, pos, neu = getPolarityOfAdjectives(set_adjectives)

'''
authorsList = set()
for citations in ALL_CITATIONS:
    temp = getAllAuthors(citations)
    authorsList.update(temp)

authorsList = list(set(authorsList))

final = []
for author in authorsList:
    h = hIndex(ALL_CITATIONS, author)
    final.append({'author': author, 'hIndex': h})
'''
# print(*final)

writeCitationsToCSV(ALL_CITATIONS)
