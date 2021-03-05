### Adapted from the documentation provided here: https://pypi.org/project/requests-html/
### Written by: Mukundan Kuthalam
from requests_html import HTMLSession
import sys
import json
import re

def openSession(url):
    request = None # HTTP Request to scrape the website source

    # Open a request to fetch the HTML content
    try:
        session = HTMLSession()
        request = session.get(url)
    except:
        request = None

    return request

def formulateJSON(request):
    finalJSON = dict() # Result that gets moved to a json
    # Keys used in the final JSON
    recipeKey = "recipeName"
    ingredientsKey = "ingredients"
    instructionsKey = "instructions"

    # Get the portion of the website source that contains the info we need
    # Notice this object comes as a string, so we need to parse it to what we want
    recipeDetails = request.html.find("script", first = True).text

    # Now we get each necessary part of info by traversing the above string
    finalJSON[recipeKey] = getRecipeName(recipeDetails)
    finalJSON[ingredientsKey] = getIngredients(recipeDetails)
    finalJSON[instructionsKey] = getInstructions(recipeDetails)

    return finalJSON

def getRecipeName(recipeInfo):
    recipeNameTag = "name"

    # Parse out the recipe name
    nameField = recipeInfo[recipeInfo.index("mainEntityOfPage"):] # Unique key to get us close to the recipe name
    nameField = nameField[nameField.index(recipeNameTag + "\":") + len(recipeNameTag) + 2:]
    nameField = nameField[:nameField.index(",")] # From the key name to the first following comma

    # Now get rid of the spaces and quotes
    nameField = re.sub("\"", "", nameField)[1:]

    return nameField

def getIngredients(recipeInfo):
    ingredientSiteTag = "recipeIngredient"

    # Parse out the section with all ingredients
    ingredientString = recipeInfo[recipeInfo.index(ingredientSiteTag):]
    ingredientString = ingredientString[ingredientString.index("["):ingredientString.index("]") + 1]

    # Construct the list of ingredients
    ingredientList = []
    ingredientsLeft = True
    while ingredientsLeft:
        # Get the ingredient phrase
        startIdx = ingredientString.index("\"") + 1
        stopIdx = ingredientString[startIdx:].index("\"") + startIdx
        ingredientList.append(ingredientString[startIdx:stopIdx].strip())

        # Any ingredients left?
        ingredientString = ingredientString[stopIdx + 1:]
        try: # Check if there is a quote left in the string
            ingredientString.index("\"")
        except:
            ingredientsLeft = False

    return ingredientList

def getInstructions(recipeInfo):
    instructionsTag = "recipeInstructions"

    # Parse out just the section with instructions
    instructionsString = recipeInfo[recipeInfo.index(instructionsTag):]
    instructionsString = instructionsString[instructionsString.index("["):instructionsString.index("]") + 1]

    # Construct the list of instructions
    instructionsList = []
    instructionsLeft = True
    while instructionsLeft:
        # Get the instruction text
        instructionsString = instructionsString[instructionsString.index("text\":") + 8:]
        stopIdx = instructionsString.index("\\")
        currentStep = instructionsString[:stopIdx]

        # Since a single step can actually contain multiple steps, some further parsing is done
        instructionCollection = currentStep.split(".")
        for elem in instructionCollection:
            furtherParsedElem = elem.split(";") # There may be semicolons that separate steps
            for item in furtherParsedElem:
                if item != "":
                    instructionsList.append(item.strip().capitalize())

        # Any instructions left?
        instructionsString = instructionsString[stopIdx + 1:]
        try:
            instructionsString.index("text\":")
        except:
            instructionsLeft = False

    return instructionsList

if __name__ == "__main__":
    request = openSession(sys.argv[1])
    print(formulateJSON(request))
