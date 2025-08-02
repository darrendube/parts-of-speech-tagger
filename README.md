![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
# Parts of Speech Tagger

POSTagger is a parts-of-speech tagger based on a Hidden Markov Model. 
Given a new sentence, it uses Viterbi decoding to compute the most likely sequence of tags for the sentence, based on transition and emission probabilities learned from training data. 

Author: [Darren Dube](https://github.com/darrendube)   
🔗      [LinkedIn](https://linkedin.com/in/darrendube)  
🌐      [darrendube.github.io](https://darrendube.github.io/)  

## Brief Overview

Parts of speech (POS) are the basic categories that we group words in. The main parts of speech in English are: noun, verb, adjective, adverb, pronoun, preposition, conjunction, and interjection. Some tagging schemes may split these categories further: the Brown Corpus (the dataset this model was tested on) has over 80 parts-of-speech. 

POS tagging is crucial in Natural Language Processing (NLP): many downstream NLP tasks rely on knowing the structure and "meaning" of text. As a result, many machine learning models have been built to assign POS tags to text. This project is one of them. 

To initialise the model, you train it using sequences of training sentences and tags. After training, you can give the model a sentence, say, 
> "The quick brown fox jumps over the lazy dog"
      
and it should output the parts of speech:         

> DETERMINER, ADJECTIVE, ADJECTIVE, NOUN, VERB, PREPOSITION, DETERMINER, ADJECTIVE, NOUN

(the exact POS tags may differ based on the specific scheme used in the training data).    

Example usage is shown in the accompanying Jupyter Notebook. 

## Dependencies

To run this file, you need:
- Python
- Numpy

## How to run

1. Clone this repository: `git clone git@github.com:darrendube/parts-of-speech-tagger.git`
2. Set up a Python environment
3. Install `numpy`
4. Start up Jupyter and follow the usage instructions in the `POSTagger` docstring
