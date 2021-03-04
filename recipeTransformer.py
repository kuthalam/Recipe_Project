# Where the Spacy code was adapted from: https://spacy.io/usage/linguistic-features

from recipeScraper import openSession, formulateJSON
import spacy
import sys
import re
import nltk
import requests
import random

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
                    "healthy": ["chicken", "turkey", "coconut oil", "poultry", "fish"], # This list was constructed while referring to https://www.heart.org/en/healthy-living/healthy-eating/eat-smart/nutrition-basics/meat-poultry-and-fish-picking-healthy-proteins
                    "unhealthy": ["steak", "beef", "sausage", "butter", "ham", "salami"],
                    "spices": ["seasoning", "oregano"],
                    "condiments": ["salt"]}
    allFoods = set()
    cookingVerbs = ["place"] # ConceptNet can be very bad at detecting what things are verbs

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

    ############################################################################
    # Name: _ingParse                                                          #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Using the recipe data store in self.recipeData, this parses out   #
    # the relevant portions (ingredient name, quantity, and measurement units) #
    # from the recipe data and saves these in the self.ingPredicates dict.     #
    # By predicate, we mean something like "(isa beef ingredient)", but to be  #
    # Pythonic (or we think we're being Pythonic), this is in a dict structure #
    # (ex. self.ingPredicates["beef"]["isa"] = "beef").                        #
    # The parsing combines using a dependency parser and conceptNet to narrow  #
    # the "isa" predicate to point to a food. Oddly enough, whether the root   #
    # is an actual food can be separate from the often correct quantity and    #
    # measurement parsing.                                                     #
    ############################################################################
    def _ingParse(self):
        for i in range(len(self.recipeData["ingredients"])): # So we can distinguish between different ingredients with the same root
            ing = self.recipeData["ingredients"][i]
            parsedText = self.nlp(ing)
            mainToken = None # The actual ingredient name
            for token in parsedText: # Construct the appropriate predicates
                if token.dep_ == "ROOT": # Now we can traverse the parse tree

                    # Now let's check if the root word is actually food
                    rootRequest = requests.get("http://api.conceptnet.io/c/en/" + token.text.lower() + "?offset=0&limit=" + str(self.queryOffset)).json()
                    if self._isAFood(token.text.lower()):
                        mainToken = token.text
                    else: # If this fails, check every word in the sentence for food
                        for newToken in parsedText:
                            if self._isAFood(newToken.text.lower()):
                                mainToken = newToken.text
                                break # No need to keep going if we've got a food, as what came before the first food term was likely adjectives

                    # Now assign values based on the ingredient name
                    if not mainToken is None:
                        dictKey = mainToken + " " + str(i) # This is the key with which all info for this ingredient can be retrieved
                        self.ingPredicates[dictKey] = dict()
                        self.ingPredicates[dictKey]["isa"] = mainToken
                        ing = ing.replace(mainToken, "isa") # Save a "cleaned" version of the sentence
                        for child in token.children: # Only related words are considered to be useful
                            requestObj = requests.get("http://api.conceptnet.io/c/en/" + child.text + "_" + mainToken + "?offset=0&limit=" + str(self.queryOffset)).json()

                            for edge in requestObj["edges"]: # Check if there are two word phrases like ground beef, so we can make a note of the entire phrase
                                eachEdge = edge["@id"].split(",") # Look for child.text + " " + token.text isa food
                                if "isa" in eachEdge[0].lower() and "/" + token.text.lower() + "/" in eachEdge[1].lower() and "food" in eachEdge[2].lower():
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

    ############################################################################
    # Name: _isAFood                                                           #
    # Params: candidate (this is the thing that you are determining is a food  #
    # or not)                                                                  #
    # Returns: Boolean                                                         #
    # Notes: Leverage ConceptNet to see if candidate is a food. Since          #
    # ConceptNet is not particularly reliable, we also made use of our         #
    # allFoods set (which was built in __init__).                              #
    ############################################################################
    def _isAFood(self, candidate):
        requestJSON = requests.get("http://api.conceptnet.io/c/en/" + candidate + "?offset=0&limit=" + str(self.queryOffset)).json()
        if candidate in self.allFoods: # First check against our set of foods
            return True
        else: # If it's not there, then see if ConceptNet calls it a food
            for edge in requestJSON["edges"]:
                eachEdge = edge["@id"].split(",")
                if "isa" in eachEdge[0].lower() and "/" + candidate.lower() + "/" in eachEdge[1].lower() and "food" in eachEdge[2].lower():
                    return True
        return False

    ############################################################################
    # Name: _instParse                                                         #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Using the recipe data store in self.recipeData, this parses out   #
    # the relevant portions (primary method and associated tool). We again     #
    # parse into predicates like before. The thing is we needed ConceptNet a   #
    # lot more here. Parsing instructions seems a lot more difficult than      #
    # parsing ingredients.                                                     #
    ############################################################################
    def _instParse(self):
        for i in range(len(self.recipeData["instructions"])):
            inst = self.recipeData["instructions"][i]
            parsedText = self.nlp(inst)
            additionalRoot = False # Turns true if the potential for a second root word pops up; semaphore to avoid storing that root word
            mainToken = None # This is the root word that turns into the primary method
            for token in parsedText:
                if token.dep_ == "ROOT" and not additionalRoot: # If you have multiple root words, go with the first one
                    additionalRoot = True # So we don't end up checking more root words - the first one (reading left-to-right) is usually what you want

                    if self._isAnAction(token.text): # Now let's check if the root word is actually a verb
                        mainToken = token.text
                    else: # If the above fails, check every word in the sentence for the first verb that is an action
                        for newToken in parsedText:
                            if self._isAnAction(newToken.text) and mainToken is None:
                                mainToken = newToken.text

                    if not mainToken is None:
                        self.instPredicates[mainToken + str(i)] = dict()
                        self.instPredicates[mainToken + str(i)]["primaryMethod"] = mainToken
                        inst = inst.replace(mainToken, "primaryMethod")
                        for child in token.children: # Now we start relying on ConceptNet to check if any of these children are cooking tools
                            requestObj = requests.get("http://api.conceptnet.io/c/en/" + child.text + "?offset=0&limit=" + str(self.queryOffset)).json()
                            for edge in requestObj["edges"]:
                                if "usedfor" in edge["@id"].lower() and edge["end"]["label"].lower() == "cook":
                                    self.instPredicates[mainToken + str(i)]["toolFor"] = child.text
                                    inst = inst.replace(child.text, "toolFor")
                        self.instPredicates[mainToken + str(i)]["sentence"] = inst

    ############################################################################
    # Name: _isAnAction                                                        #
    # Params: candidate (this is the thing that you are determining is a verb).#
    # Returns: Boolean                                                         #
    # Notes: Similar to _isAFood. This is just for the primaryMethod and       #
    # checks for verbs.                                                        #
    ############################################################################
    def _isAnAction(self, candidate):
        requestJSON = requests.get("http://api.conceptnet.io/c/en/" + candidate.lower() + "?offset=0&limit=" + str(self.queryOffset)).json()
        if candidate.lower() in self.cookingVerbs: # Since ConceptNet can be bad at detecting what is a verb
            return True
        else:
            for edge in requestJSON["edges"]:
                eachEdge = edge["@id"].split(",") # Check if this word is ever used as a verb
                if "mannerof" in eachEdge[0].lower() and ("/" + candidate.lower() + "/v/" in eachEdge[1].lower() or \
                "/" + candidate.lower() + "/v/" in eachEdge[2].lower()): # If the found root word is a verb
                    return True
        return False

    ############################################################################
    # Name: _decideTransformation                                              #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Short helper to get user input on how to transform the input      #
    # recipe.                                                                  #
    ############################################################################
    def _decideTransformation(self):
        self.transformationType = input("""What would you like us to transform the given dish into?\n\nPlease enter one of the options below:
        - \"to vegetarian\" or \"from vegetarian\"
        - \"to healthy\" or \"from healthy\"
        - \"to <insert cuisine here>\"\nEnter choice here: """)

        allTypes = ["to vegetarian", "to healthy", "from vegetarian", "from healthy", "to Mexican"]

        while not self.transformationType in allTypes:
            print("\nI'm sorry, it looks like that was not a valid transformation. Could you please review the list of transformations and input again? We do need you to input the exact phrases above: ")
        print("\nSo we are going to be transforming " + self.recipeData["recipeName"] + " in accordance with the \"" + self.transformationType + "\" option.")

    ############################################################################
    # Name: _ingTransformation                                                 #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Transform the ingredients in accordance with the transformation   #
    # the user specified. We also store a mapping between the old ingredients  #
    # and their transformed values to make the instruction transformation      #
    # easier (this is what self.transformedIng is for).                        #
    ############################################################################
    def _ingTransformation(self):
        if self.transformationType == "to vegetarian":
            for ing in self.ingPredicates.keys():
                allRelevantPred = self.ingPredicates[ing]
                finalSent = allRelevantPred["sentence"]
                if any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # If you found a meat protein
                    finalSent = finalSent.replace("isa", self.replacementGuide["vegProtein"][0]) # Then tofu is pretty much the go-to replacement
                    self.transformedIng[allRelevantPred["isa"]] = self.replacementGuide["vegProtein"][0] # Keep track of the transformed ingredients
                else: # If the ingredient's not a meat protein, there's nothing to replace
                    finalSent = finalSent.replace("isa", allRelevantPred["isa"])
                if "quantity" in allRelevantPred.keys():
                    finalSent = finalSent.replace("quantity", allRelevantPred["quantity"])
                if "measurement" in allRelevantPred.keys():
                    finalSent = finalSent.replace("measurement", allRelevantPred["measurement"])
                self.finalIng.append(finalSent)

        elif self.transformationType == "to healthy": # The use of "any" below is due to ingredient names like "ground beef" where "beef" is the only word in our dict
            for ing in self.ingPredicates.keys():
                allRelevantPred = self.ingPredicates[ing]
                finalSent = allRelevantPred["sentence"]
                if any([item in self.replacementGuide["unhealthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # Found an unhealthy meat protein
                    finalSent = finalSent.replace("isa", "chicken")
                    self.transformedIng[allRelevantPred["isa"]] = "chicken" # Keep track of the transformed ingredients
                elif any([item in self.replacementGuide["unhealthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                not any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # Found an unhealthy non-meat protein (butter for us)
                    finalSent = finalSent.replace("isa", "coconut oil")
                    self.transformedIng[allRelevantPred["isa"]] = "coconut oil" # Keep track of the transformed ingredients
                else: # This is not an ingredient that we need to replace for health reasons
                    finalSent = finalSent.replace("isa", allRelevantPred["isa"])
                if "quantity" in allRelevantPred.keys():
                    finalSent = finalSent.replace("quantity", allRelevantPred["quantity"])
                if "measurement" in allRelevantPred.keys():
                    finalSent = finalSent.replace("measurement", allRelevantPred["measurement"])
                self.finalIng.append(finalSent)

        elif self.transformationType == "from healthy":
            for ing in self.ingPredicates.keys():
                allRelevantPred = self.ingPredicates[ing]
                finalSent = allRelevantPred["sentence"]
                if any([item in self.replacementGuide["healthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # If we found a healthy meat
                    meatReplacementOptions = list(set(self.replacementGuide["meatProtein"]).intersection(set(self.replacementGuide["unhealthy"])))
                    replaceWith = meatReplacementOptions[random.randrange(len(meatReplacementOptions))] # Make a random unhealthy meat the replacement
                    finalSent = finalSent.replace("isa", replaceWith)
                    self.transformedIng[allRelevantPred["isa"]] = replaceWith # Keep track of the transformed ingredients
                elif any([item in self.replacementGuide["healthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                not any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # Found a healthy non-meat? (No use for now)
                    print("No non-meat unhealthy substitutes.") # This is just a placeholder for potential future work
                else: # This is not an ingredient that we need to replace for health reasons
                    finalSent = finalSent.replace("isa", allRelevantPred["isa"])
                if "quantity" in allRelevantPred.keys():
                    finalSent = finalSent.replace("quantity", allRelevantPred["quantity"])
                if "measurement" in allRelevantPred.keys():
                    finalSent = finalSent.replace("measurement", allRelevantPred["measurement"])
                self.finalIng.append(finalSent)
        else: # Placeholder for now
            print("\nValid option - we just haven't gotten to it yet. Sorry!")
            sys.exit(0)

    ############################################################################
    # Name: _instTransformation                                                #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Using the transformed ingredients from before                     #
    # (self.transformedIng), we transform the ingredients into their           #
    # appropriate versions.                                                    #
    ############################################################################
    def _instTransformation(self):
        for inst in self.instPredicates.keys():
            origInst = self.instPredicates[inst]["sentence"] # This is the sentence that goes through the cascade of transformations
            newInst = None

            # First replace the ingredient if needed
            for oldIng in self.transformedIng:
                if oldIng in origInst or any([x in origInst for x in oldIng.split(" ")]):
                    if len(oldIng) > 1: # If the "any" part of the above clause was what caused the condition to trigger
                        # Then we need to iterate over those values to make sure everything in the old instruction is properly replaced
                        for ing in oldIng.split(" "):
                            newInst = origInst.replace(ing, self.transformedIng[oldIng])
                    else:
                        newInst = origInst.replace(ing, self.transformedIng[oldIng])
                    self.transformedInst[origInst] = newInst # "Backpointer" of sorts

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

    ############################################################################
    # Name: _printNewIngredients                                               #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Pretty print the transformed ingredients. Putting this            #
    # documentation block here for consistency.                                #
    ############################################################################
    def _printNewIngredients(self):
        print("\nYour new ingredients are: ")
        for ing in self.finalIng:
            print("- " + ing)

    ############################################################################
    # Name: _printNewInstructions                                              #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Pretty print the transformed instructions. Putting this           #
    # documentation block here for consistency.                                #
    ############################################################################
    def _printNewInstructions(self):
        print("\nYour new instructions are: ")
        for i in range(len(self.finalInst)):
            print(str(i + 1) + ". " + self.finalInst[i])

    ############################################################################
    # Name: transform                                                          #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: This is really the entire spindle (i.e. function that ties        #
    # everything together). Putting this documentation block here for          #
    # consistency.                                                             #
    ############################################################################
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
    userRecipeURL = input("\nHello and welcome to the recipe transformer! Please enter an AllRecipes URL that gives us a recipe to transform: ")
    newTransformer = Transformer(userRecipeURL.strip())
    print("\nThank you! We will be asking for more input momentarily, so please wait as we get everything ready (this could take a while).")
    newTransformer.transform()
