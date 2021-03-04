# Where the Spacy code was adapted from: https://spacy.io/usage/linguistic-features

from recipeScraper import openSession, formulateJSON
import spacy
import sys
import re
import nltk
import requests

class Transformer:
    recipeData = None

    ingPredicates = dict()
    instPredicates = dict()

    transformedIng = dict()
    transformedInst = dict()

    finalIng = list()
    finalInst = list()

    nlp = spacy.load("en_core_web_sm")

    replacementGuide = {"vegProtein": ["tofu"],
                    "meatProtein": ["beef", "chicken", "pork", "pepperoni", "sausage", "turkey",
                    "steak", "fish", "salmon", "shrimp", "lobster", "salami", "rennet", "poultry"],
                    "egg": ["veganEgg"], # Special case
                    "veganEgg": ["egg"], # Special case v2
                    "standardDairy": ["milk", "cheese", "cream", "yogurt", "butter", "ghee"],
                    "healthy": ["chicken", "turkey", "coconut oil"],
                    "unhealthy": ["steak", "beef", "sausage", "butter"],
                    "spices": ["seasoning", "oregano"],
                    "condiments": ["salt"]}
    allFoods = set()

    userPrompt = """Hello! Welcome to the recipe transformer. We noticed you've given us a dish already. What would you like us to transform it into?\n\nPlease enter one of the options below:
    - \"to vegetarian\" or \"from vegetarian\"
    - \"to healthy\" or \"from healthy\"
    - \"to <insert cuisine here>\"\nEnter choice here: """

    transformationType = None
    queryOffset = 100 # Number of results returned by ConceptNet API call

    ############################################################################
    # Name: __init__                                                           #
    # Params: url (the url from which a recipe is fetched)                     #
    # Returns: None                                                            #
    # Notes: Makes a HTTP request to get the right information about the       #
    # recipe and format it into a JSON. Also constructs a set of foods we know #
    # about for later convenience.                                             #
    ############################################################################
    def __init__(self, url):
        request = openSession(url)
        self.recipeData = formulateJSON(request)

        for key in self.replacementGuide.keys():
            for food in self.replacementGuide[key]:
                self.allFoods.add(food)

    def _ingParse(self):
        for i in range(len(self.recipeData["ingredients"])): # So we can distinguish between different ingredients with the same root
            ing = self.recipeData["ingredients"][i]
            parsedText = self.nlp(ing)
            mainToken = None # The actual ingredient name
            for token in parsedText: # Construct the appropriate predicates
                if token.dep_ == "ROOT": # Now we can traverse the parse tree

                    # Now let's check if the root word is actually food
                    rootRequest = requests.get("http://api.conceptnet.io/c/en/" + token.text + "?offset=0&limit=" + str(self.queryOffset)).json()
                    if token.text in self.allFoods: # First check against our set of foods
                        mainToken = token.text
                    elif mainToken == None: # No need to keep going if we've got a food, as what came before the first food term was likely adjectives
                        for edge in rootRequest["edges"]:
                            isaEdge = edge["@id"].split(",") # Look for token.text isa food
                            if "isa" in isaEdge[0].lower() and "/" + token.text.lower() + "/" in isaEdge[1].lower() and "food" in isaEdge[2].lower():
                                mainToken = token.text
                    if mainToken == None: # If this fails, check every word in the sentence for food
                        for newToken in parsedText:
                            newRequest = requests.get("http://api.conceptnet.io/c/en/" + newToken.text + "?offset=0&limit=" + str(self.queryOffset)).json()
                            if newToken.text in self.allFoods:
                                mainToken = newToken.text
                            elif mainToken == None: # No need to keep going if we've got a food, as what came before the first food term was likely adjectives
                                for edge in newRequest["edges"]:
                                    isaEdge = edge["@id"].split(",") # Look for token.text isa food
                                    if "isa" in isaEdge[0].lower() and "/" + newToken.text.lower() + "/" in isaEdge[1].lower() and "food" in isaEdge[2].lower():
                                        mainToken = newToken.text

                    # Now assign values based on the ingredient name
                    if not mainToken is None:
                        dictKey = mainToken + " " + str(i) # This is the key with which all info for this ingredient can be retrieved
                        self.ingPredicates[dictKey] = dict()
                        self.ingPredicates[dictKey]["isa"] = mainToken
                        ing = ing.replace(mainToken, "isa") # Save a "cleaned" version of the sentence
                        for child in token.children: # Only related words are considered to be useful
                            requestObj = requests.get("http://api.conceptnet.io/c/en/" + child.text + "_" + mainToken + "?offset=0&limit=" + str(self.queryOffset)).json()
                            if len(requestObj["edges"]) != 0:
                                if "end" in requestObj["edges"][0].keys():
                                    if "sense_label" in requestObj["edges"][0]["end"].keys():
                                        if "food" in requestObj["edges"][0]["end"]["sense_label"].lower():
                                            ing = ing.replace("isa", mainToken) # Get back the original sentence
                                            self.ingPredicates[dictKey]["isa"] = child.text + " " + mainToken
                                            ing = ing.replace(child.text + " " + mainToken, "isa")
                            if any([x.text.isdigit() for x in child.children]): # The measurement and amount, tied together by the parser
                                for item in child.children:
                                    if item.text.isdigit():
                                        self.ingPredicates[dictKey]["quantity"] = item.text
                                        self.ingPredicates[dictKey]["measurement"] = child.text
                                        ing = ing.replace(item.text, "quantity") # For later use
                                        ing = ing.replace(child.text, "measurement") # For later use
                        self.ingPredicates[dictKey]["sentence"] = ing

    def _instParse(self):
        for i in range(len(self.recipeData["instructions"])):
            inst = self.recipeData["instructions"][i]
            parsedText = self.nlp(inst)
            additionalRoot = False # Turns true if the potential for a second root word pops up; semaphore to avoid storing that root word
            for token in parsedText:
                if token.dep_ == "ROOT" and not additionalRoot: # If you have multiple root words, go with the first one
                    additionalRoot = True
                    self.instPredicates[token.text + str(i)] = dict()
                    self.instPredicates[token.text + str(i)]["primaryMethod"] = token.text
                    inst = inst.replace(token.text, "primaryMethod")
                    for child in token.children: # Now we start relying on ConceptNet to check if any of these children are cooking tools
                        requestObj = requests.get("http://api.conceptnet.io/c/en/" + child.text + "?offset=0&limit=" + str(self.queryOffset)).json()
                        for edge in requestObj["edges"]:
                            if "usedfor" in edge["@id"].lower() and edge["end"]["label"].lower() == "cook":
                                self.instPredicates[token.text + str(i)]["toolFor"] = child.text
                                inst = inst.replace(child.text, "toolFor")
                    self.instPredicates[token.text + str(i)]["sentence"] = inst

    def _decideTransformation(self):
        self.transformationType = input(self.userPrompt)

        print("\nSo we are going to be transforming " + self.recipeData["recipeName"] + " " + self.transformationType + " dish.")

    def _ingTransformation(self):
        if self.transformationType == "to vegetarian":
            for ing in self.ingPredicates.keys():
                allRelevantPred = self.ingPredicates[ing]
                finalSent = allRelevantPred["sentence"]
                if any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]):
                    finalSent = finalSent.replace("isa", self.replacementGuide["vegProtein"][0])
                    self.transformedIng[allRelevantPred["isa"]] = self.replacementGuide["vegProtein"][0] # Keep track of the transformed ingredients
                else:
                    finalSent = finalSent.replace("isa", allRelevantPred["isa"])
                if "quantity" in allRelevantPred.keys():
                    finalSent = finalSent.replace("quantity", allRelevantPred["quantity"])
                if "measurement" in allRelevantPred.keys():
                    finalSent = finalSent.replace("measurement", allRelevantPred["measurement"])
                self.finalIng.append(finalSent)
        elif self.transformationType == "to healthy":
            for ing in self.ingPredicates.keys():
                allRelevantPred = self.ingPredicates[ing]
                finalSent = allRelevantPred["sentence"]
                if any([item in self.replacementGuide["unhealthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]):
                    finalSent = finalSent.replace("isa", "chicken")
                    self.transformedIng[allRelevantPred["isa"]] = "chicken" # Keep track of the transformed ingredients
                elif (any([item in self.replacementGuide["unhealthy"] for item in allRelevantPred["isa"].split(" ")]) \
                or allRelevantPred["isa"].split(" ")[-1] in self.replacementGuide["unhealthy"]) \
                and not any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # To handle multi-word "isa"s
                    finalSent = finalSent.replace("isa", "coconut oil")
                    self.transformedIng[allRelevantPred["isa"]] = "coconut oil" # Keep track of the transformed ingredients
                else:
                    finalSent = finalSent.replace("isa", allRelevantPred["isa"])
                if "quantity" in allRelevantPred.keys():
                    finalSent = finalSent.replace("quantity", allRelevantPred["quantity"])
                if "measurement" in allRelevantPred.keys():
                    finalSent = finalSent.replace("measurement", allRelevantPred["measurement"])
                self.finalIng.append(finalSent)

    def _instTransformation(self):
        for inst in self.instPredicates.keys():
            origInst = self.instPredicates[inst]["sentence"] # This is the sentence that goes through the cascade of transformations
            newInst = None

            # First replace the ingredient if needed
            for oldIng in self.transformedIng:
                if oldIng in self.instPredicates[inst]["sentence"] or \
                any([x in self.instPredicates[inst]["sentence"] for x in oldIng.split(" ")]):
                    if len(oldIng) > 1: # If the "any" part of the above clause was what caused the condition to trigger
                        # Then we need to iterate over those values to make sure everything in the old instruction is properly replaced
                        for ing in oldIng.split(" "):
                            newInst = origInst.replace(ing, self.transformedIng[oldIng])
                    else:
                        newInst = origInst.replace(ing, self.transformedIng[oldIng])
                    self.transformedInst[self.instPredicates[inst]["sentence"]] = newInst # "Backpointer" of sorts

            # Now substitute the primary method
            if newInst is None:
                newInst = origInst.replace("primaryMethod", self.instPredicates[inst]["primaryMethod"])
            else:
                newInst = newInst.replace("primaryMethod", self.instPredicates[inst]["primaryMethod"])

            # Substituting the tool used is even easier now, since newInst will never be None
            if "toolFor" in self.instPredicates[inst].keys():
                newInst = newInst.replace("toolFor", self.instPredicates[inst]["toolFor"])

            # Time to collect the new instruction
            self.finalInst.append(newInst)

    def _printNewIngredients(self):
        print("\nYour new ingredients are: ")
        for ing in self.finalIng:
            print(ing)

    def _printNewInstructions(self):
        print("\nYour new instructions are: ")
        for i in range(len(self.finalInst)):
            print(str(i + 1) + ". " + self.finalInst[i])

    def transform(self):
        # First build the data structures
        newTransformer._ingParse()
        newTransformer._instParse()

        # Now we alert the user to what they decided to do
        self._decideTransformation()

        self._ingTransformation()
        self._instTransformation()

        self._printNewIngredients()
        self._printNewInstructions()

if __name__ == "__main__":
    # Example command: python recipeTransformer.py https://www.allrecipes.com/recipe/21242/pizza-pasta/
    newTransformer = Transformer(sys.argv[1])
    newTransformer.transform()
