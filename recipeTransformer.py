# Where the Spacy code was adapted from: https://spacy.io/usage/linguistic-features

from recipeScraper import openSession, formulateJSON
import spacy
import sys
import re
import nltk

class Transformer:
    recipeData = None
    ingPredicates = list()
    instPredicates = list()
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, url):
        request = openSession(url)
        self.recipeData = formulateJSON(request)

    def ingParse(self):
        for ing in self.recipeData["ingredients"]:
            parsedText = self.nlp(ing)
            for token in parsedText: # Construct the appropriate predicates
                if token.dep_ == "ROOT": # Now we can traverse the parse tree
                    self.ingPredicates.append("(isa " + token.text + " ingredient)")
                    ing = ing.replace(token.text, "_") # Save a "cleaned" version of the sentence
                    for child in token.children: # For each of its children
                        if any([x.text.isdigit() for x in child.children]): # Now we have an amount and measurement
                            for item in child.children:
                                if item.text.isdigit():
                                    self.ingPredicates.append("(quantity " + token.text + " " + item.text + ")")
                                    self.ingPredicates.append("(measurement " + token.text + " " + child.text + ")")
                                    ing = ing.replace(item.text, "_") # For later use
                                    ing = ing.replace(child.text, "_") # For later use
                    self.ingPredicates.append("(sentence " + token.text + " " + ing + ")")

    def instParse(self):
        for inst in self.recipeData["instructions"]:
            parsedText = self.nlp(inst)
            for token in parsedText:
                if token.dep_ == "ROOT":
                    self.instPredicates.append("(isa " + token.text + " primaryMethod)")
                    inst = inst.replace(token.text, "_")
                    for child in token.children:
                        if len(child.text) > 1 and not child.is_stop: # Make sure you're getting an actual tool.
                            self.instPredicates.append("(toolFor " + token.text + " " + child.text + ")")
                            inst = inst.replace(child.text, "_")
                    self.instPredicates.append("(sentence " + token.text + " " + inst + ")")

    def prettyPrintIng(self):
        print("Ingredient propositions:")
        for ing in self.ingPredicates:
            if "isa" in ing: # We've encountered a new ingredient
                print()
                print("For " + re.findall("\w+", ing.replace("isa ", ""))[0] + ":")
            print(ing)

    def prettyPrintInst(self):
        print("Instruction propositions:")
        for inst in self.instPredicates:
            if "isa" in inst: # We've encountered a new instruction
                print()
                print("For " + re.findall("\w+", inst.replace("isa ", ""))[0] + ":")
            print(inst)

    def transformPrep(self):
        newTransformer.ingParse()
        newTransformer.instParse()

if __name__ == "__main__":
    # Example command: python recipeTransformer.py https://www.allrecipes.com/recipe/21242/pizza-pasta/
    newTransformer = Transformer(sys.argv[1])
    newTransformer.transformPrep()

    # Pretty print the internal data structures
    newTransformer.prettyPrintIng()
    newTransformer.prettyPrintInst()
