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

    finalIng = list()
    finalInst = list()

    nlp = spacy.load("en_core_web_sm")

    replacementGuide = {"vegProtein": ["tofu"],
                    "meatProtein": ["beef", "chicken", "pork", "pepperoni", "sausage", "turkey",
                    "steak", "fish", "salmon", "shrimp", "lobster", "salami", "rennet", "poultry",
                    "bacon", "lamb"],
                    "pairedWords": ["stock", "broth", "sauce", "loin", "tenderloin", "sirloin", "breast"],
                    "worcestershire sauce": ["soy sauce"],
                    "standardDairy": ["milk", "cheese", "cream", "yogurt", "butter", "ghee"],
                    "healthy": ["chicken", "turkey", "coconut oil", "poultry", "fish"], # This list was constructed by referring to https://www.heart.org/en/healthy-living/healthy-eating/eat-smart/nutrition-basics/meat-poultry-and-fish-picking-healthy-proteins
                    "unhealthy": ["steak", "beef", "sausage", "butter", "ham", "salami", "bacon", "sirloin"],
                    "spices": ["seasoning", "oregano"],
                    "condiments": ["salt", "oil"],
                    "plants": ["onions", "onion"]}
    allFoods = set()
    cookingVerbs = ["place"] # ConceptNet can be very bad at detecting what things are verbs

    spicesForStyleReplacement = set()
    styleReplacementGuide = {"Mexican": {"sausage": "chorizo", "cheese": "queso fresco",
    "spices": ["Chili Powder", "Cilantro", "Coriander", "Cumin", "Garlic Powder", "Onion Powder", "Smoked Paprika"]}}

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
    # Pythonic, this is in a dict structure                                    #
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
            additionalRoot = False # Turns true if the potential for a second root word pops up; semaphore to avoid storing that root word
            mainToken = None # The actual ingredient name
            for token in parsedText: # Construct the appropriate predicates
                if token.dep_ == "ROOT" and not additionalRoot: # Now we can traverse the parse tree
                    additionalRoot = True

                    # Now let's check if the root word is actually food
                    if self._isAFood(token.text.lower()):
                        mainToken = token.text
                    else: # If this fails, check every word in the sentence for food
                        for newToken in parsedText:
                            if self._isAFood(newToken.text.lower()):
                                mainToken = newToken.text
                                break # No need to keep going if we've got a food, as what came before the first food term was likely adjectives

                    # Sometimes, you get None for odd reasons. Seems better to go with what we have rather than adding an obscure layer of parsing.
                    # That said, there is one last check after this
                    if mainToken is None:
                        mainToken = token.text

                    # If you get beef sirloin, pork loin/tenderloin, or a kind of stock or broth,
                    # replace both words (e.g. chicken broth), not just "sirloin" or "broth"
                    if mainToken in self.replacementGuide["pairedWords"]:
                        mainToken = ing[ing.index(token) - 1] + " " + token.text

                    # Now assign values based on the ingredient name
                    dictKey = mainToken + " " + str(i) # This is the key with which all info for this ingredient can be retrieved
                    self.ingPredicates[dictKey] = dict()
                    self.ingPredicates[dictKey]["isa"] = mainToken
                    ing = ing.replace(mainToken, "isa") # Save a "cleaned" version of the sentence
                    for child in token.children: # Only related words are considered to be useful
                        requestObj = requests.get("http://api.conceptnet.io/c/en/" + child.text.lower() + "_" + mainToken.lower() + "?offset=0&limit=" + str(self.queryOffset)).json()

                        for edge in requestObj["edges"]: # Check if there are two word phrases like ground beef, so we can make a note of the entire phrase
                            eachEdge = edge["@id"].split(",") # Look for child.text + " " + token.text isa food
                            if "isa" in eachEdge[0].lower() and "/" + child.text.lower() + "_" + mainToken.lower() + "/" in eachEdge[1].lower() and "food" in eachEdge[2].lower():
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
    # allFoods set (which was built in __init__). We also collect spices for   #
    # the transformation to another cuisine.                                   #
    ############################################################################
    def _isAFood(self, candidate):
        requestJSON = requests.get("http://api.conceptnet.io/c/en/" + candidate + "?offset=0&limit=" + str(self.queryOffset)).json()
        finalVerdict = False # Is the ingredient a food or not
        if candidate in self.allFoods: # First check against our set of foods
            finalVerdict = True
        else: # If it's not there, then see if ConceptNet calls it a food
            for edge in requestJSON["edges"]:
                eachEdge = edge["@id"].split(",")
                if "isa" in eachEdge[0].lower() and "/" + candidate.lower() + "/" in eachEdge[1].lower() and "/food" in eachEdge[2].lower():
                    finalVerdict = True
                if "isa" in eachEdge[0].lower() and "/" + candidate.lower() + "/" in eachEdge[1].lower() and "/spice" in eachEdge[2].lower():
                    self.spicesForStyleReplacement.add(candidate)
        return finalVerdict

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

                    # Sometimes, you get None for odd reasons. Seems better to go with what we have rather than adding an obscure layer of parsing
                    if mainToken is None:
                        mainToken = token.text

                    # Now we can assign the primary method and get a cooking tool for it
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
        - \"to Mexican\"\nEnter choice here: """).lower()

        allTypes = ["to vegetarian", "to healthy", "from vegetarian", "from healthy", "to mexican"]

        while not self.transformationType in allTypes:
            self.transformationType = input("\nI'm sorry, it looks like that was not a valid transformation. Could you please review the list of transformations and input again? We do need you to input the exact phrases above: ").lower()
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
        for ing in self.ingPredicates.keys():
            allRelevantPred = self.ingPredicates[ing]
            finalSent = allRelevantPred["sentence"]

            # Now for types of transformations
            if self.transformationType == "to vegetarian": # The use of "any" below is due to ingredient names like "ground beef" where "beef" is the only word in our dict
                if any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # If you found a meat protein
                    finalSent = finalSent.replace("isa", self.replacementGuide["vegProtein"][0]) # Then tofu is pretty much the go-to replacement
                    self.transformedIng[allRelevantPred["isa"]] = self.replacementGuide["vegProtein"][0] # Keep track of the transformed ingredients
                elif any([x in self.replacementGuide["pairedWords"] for x in allRelevantPred["isa"].split(" ")]): # Quick substitution for the liquids
                    if not "sauce" in allRelevantPred["isa"]: # Sauce is a bit of a special case
                        finalSent = finalSent.replace("isa", "vegetable broth") # Then vegetable broth is pretty much the go-to replacement (reference: https://www.myfrugalhome.com/broth-substitutes/)
                        self.transformedIng[allRelevantPred["isa"]] = "vegetable broth" # Keep track of the transformed ingredients
                    else:
                        finalSent = finalSent.replace("isa", "soy sauce") # Soy sauce seems reasonable (reference: https://food52.com/blog/24403-best-worcestershire-sauce-substitutes)
                        self.transformedIng[allRelevantPred["isa"]] = "soy sauce" # Keep track of the transformed ingredients

            elif self.transformationType == "to healthy":
                if any([item in self.replacementGuide["unhealthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # Found an unhealthy meat protein
                    meatReplacementOptions = list(set(self.replacementGuide["meatProtein"]).intersection(set(self.replacementGuide["healthy"])))
                    replaceWith = meatReplacementOptions[random.randrange(len(meatReplacementOptions))] # Make a random healthy meat the replacement
                    finalSent = finalSent.replace("isa", replaceWith)
                    self.transformedIng[allRelevantPred["isa"]] = replaceWith # Keep track of the transformed ingredients
                elif any([item in self.replacementGuide["unhealthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                not any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # Found an unhealthy non-meat protein (butter for us)
                    finalSent = finalSent.replace("isa", "coconut oil")
                    self.transformedIng[allRelevantPred["isa"]] = "coconut oil" # Keep track of the transformed ingredients

            elif self.transformationType == "from healthy":
                if any([item in self.replacementGuide["healthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # If we found a healthy meat
                    meatReplacementOptions = list(set(self.replacementGuide["meatProtein"]).intersection(set(self.replacementGuide["unhealthy"])))
                    replaceWith = meatReplacementOptions[random.randrange(len(meatReplacementOptions))] # Make a random unhealthy meat the replacement
                    finalSent = finalSent.replace("isa", replaceWith)
                    self.transformedIng[allRelevantPred["isa"]] = replaceWith # Keep track of the transformed ingredients
                elif any([item in self.replacementGuide["healthy"] for item in allRelevantPred["isa"].split(" ")]) and \
                not any([x in self.replacementGuide["meatProtein"] for x in allRelevantPred["isa"].split(" ")]): # Found a healthy non-meat? (No use for now)
                    print("No non-meat unhealthy substitutes.") # This is a just-in-case
                    finalSent = finalSent.replace("isa", allRelevantPred["isa"])

            elif self.transformationType == "from vegetarian":
                if any([item == "tofu" for item in allRelevantPred["isa"].split(" ")]): # If tofu ever shows up, replace it with a meat protein
                    finalSent = finalSent.replace("isa", "chicken") # Chicken works for pretty much anywhere tofu would show up
                    self.transformedIng[allRelevantPred["isa"]] = "chicken" # Keep track of the transformed ingredients

            elif self.transformationType == "to Mexican":
                if any([item in self.styleReplacementGuide["Mexican"] for item in allRelevantPred["isa"].split(" ")]): # If any of these are to be replaced
                    for item in allRelevantPred["isa"].split(" "):
                        if item in self.styleReplacementGuide["Mexican"]: # Because we need the right key (just "cheese", not "grated cheese")
                            finalSent = finalSent.replace("isa", self.styleReplacementGuide["Mexican"][item]) # Replace the ingredient with its Mexican equivalent
                            self.transformedIng[allRelevantPred["isa"]] = self.styleReplacementGuide["Mexican"][item] # Keep track of the transformed ingredients
                elif any([item in self.spicesForStyleReplacement for item in allRelevantPred["isa"].split(" ")]): # If any of these are a spice to be replaced
                    replacementSpice = self.styleReplacementGuide["Mexican"]["spices"][random.randrange(len(self.styleReplacementGuide["Mexican"]["spices"]))] # Pick a random replacement spice
                    finalSent = finalSent.replace("isa", replacementSpice) # Replace the ingredient with its Mexican equivalent spice
                    self.transformedIng[allRelevantPred["isa"]] = replacementSpice # Keep track of the transformed ingredients

            if allRelevantPred["isa"] not in self.transformedIng.keys(): # If the ingredient was not replaced, then leave it alone
                finalSent = finalSent.replace("isa", allRelevantPred["isa"])
            if "quantity" in allRelevantPred.keys():
                finalSent = finalSent.replace("quantity", allRelevantPred["quantity"])
            if "measurement" in allRelevantPred.keys():
                finalSent = finalSent.replace("measurement", allRelevantPred["measurement"])

            # Now we can collect the finalized instruction
            self.finalIng.append(finalSent)

    ############################################################################
    # Name: _instTransformation                                                #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Using the transformed ingredients from before                     #
    # (self.transformedIng), we transform the ingredients into their           #
    # appropriate versions within the instructions.                            #
    ############################################################################
    def _instTransformation(self):
        for inst in self.instPredicates.keys():
            finalInst = self.instPredicates[inst]["sentence"] # This is the sentence that goes through the cascade of transformation

            # First replace the ingredient if needed
            # Now to make the code easier and less subtle, there are two loops
            for oldIng in self.transformedIng:
                # If an entire phrase is in the original instruction and it is something that needs to be replaced
                if len(oldIng.split(" ")) > 1 and all([x.lower() in finalInst.lower() for x in oldIng.split(" ")]):
                    finalInst = finalInst.replace(oldIng, self.transformedIng[oldIng])
                    finalInst = finalInst.replace(oldIng.lower(), self.transformedIng[oldIng])

            # Now we handle individual words that show up in the original
            for oldIng in self.transformedIng:
                if oldIng.lower() in finalInst.lower() or any([x.lower() in finalInst.lower() for x in oldIng.split(" ")]):
                    if len(oldIng) > 1: # If the "any" part of the above clause was what caused the condition to trigger
                        # Then we need to iterate over those values to make sure any mention of that ingredient gets replaced
                        for ing in oldIng.split(" "): # You may have an uppercase or lowercase version of the ingredient in the sentence
                            if ing not in self.transformedIng[oldIng]: # Avoid extra replacing
                                finalInst = finalInst.replace(ing, self.transformedIng[oldIng])
                                finalInst = finalInst.replace(ing.lower(), self.transformedIng[oldIng])
                    else:
                        finalInst = finalInst.replace(oldIng, self.transformedIng[oldIng])
                        finalInst = finalInst.replace(oldIng.lower(), self.transformedIng[oldIng])

            # A note on why there are two loops above:
            # There are cases where if you mention "beef stock", you would get "beef" replaced (as an individual word), and then "stock" replaced.
            # The appropriate placement is tofu, but the instruction would now become "...tofu tofu..."
            # Thus, instead of writing potentially subtle and hard to debug code, two loops were written. The first catches "chunks" like
            # "beef stock", while the second then takes care of stragglers, so to speak.

            # Get rid of the word meat if we're transforming to a vegetarian recipe
            if self.transformationType == "to vegetarian" and "meat" in finalInst:
                finalInst = finalInst.replace("meat ", "")

            # Now substitute the primary method
            finalInst = finalInst.replace("primaryMethod", self.instPredicates[inst]["primaryMethod"])

            # Substituting the tool used, though we need to make sure it was found first
            if "toolFor" in self.instPredicates[inst].keys():
                finalInst = finalInst.replace("toolFor", self.instPredicates[inst]["toolFor"])

            # Now we put back together the "cleaned" instruction
            self.finalInst.append(finalInst)

        # If we are doing a style transformation, there's a small extra step
        if (self.transformationType == "to mexican"):
            self._instTransformationForStyle()

    ############################################################################
    # Name: _instTransformationForStyle                                        #
    # Params: None                                                             #
    # Returns: None                                                            #
    # Notes: Check if any of the style transformation spices ended up not      #
    # getting mentioned in the ingredients. If they did, do nothing, else      #
    # add an extra step to toss those in.                                      #
    ############################################################################
    def _instTransformationForStyle(self):
        for spice in self.styleReplacementGuide["Mexican"]["spices"]:
            alreadyAdded = False

            # Check if the spice was already substituted in
            for inst in self.finalInst:
                if spice.lower() in inst.lower():
                    alreadyAdded = True

            # If not, toss it in there
            if not alreadyAdded:
                self.finalIng.append(spice.capitalize()) # So only the first word is capitalized
                self.finalInst.append("Toss in some " + spice.lower() + " also")

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
