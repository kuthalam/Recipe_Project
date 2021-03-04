# Recipe_Project

Currently, you can do everything with this command: `python recipeTransformer.py`.

* Example command: `python recipeTransformer.py`
* The CLI should hopefully be able to guide the user from there.

# General Overview of Transformation Ideas/Methods
- Our parsing does the required transformations: to vegetarian, from vegetarian, to healthy, from healthy, and to Mexican.
- The following gives some general details as the kind of ways we tried making transformations:
    - To vegetarian:
        - We feel that there are many subtleties to think about like “do we consider whether vegetarians eat eggs or not?”, etc.
        - To avoid coming up with some really strange recipe results (transformed scrambled eggs, for example), we simply chose to replace the main source of protein.
        - Thus, just about any source of meat gets transformed to tofu while the general procedure stays the same.
        - The output instructions may seem a little strange at first, but generally the cooking instructions for meat stay consistent with tofu.
    - From vegetarian:
        - TODO
    - To healthy:
        - By healthy, we mainly look to sources of protein and fat. Butter and red meat like beef are examples
        - Coconut oil is generally a good source of replacement for butter, so we go with that
        - In addition, the red meats get replaced with healthier alternatives (like chicken and turkey)
    - From healthy:
        - Basically like the above “to healthy”, but replace the healthy meats with less healthy alternatives
    - To Mexican:
        - TODO

# Some Notes About Parsing
* ConceptNet tends to be inconsistent with its information. We would have to parse for very specific keys to try and get information and even then, the information was not guaranteed to be found with that key for every relevant word. You can’t even be sure that the plural of a word or phrase would be in ConceptNet when the singular is there (try sesame seed vs. sesame seeds).
* Relying on just a dependency parser is not a great idea. As we found in Project 1, parsing needs multiple layers.
* Thus, when we parsed out information from ingredients and instructions, we also made use of the hard-coded dictionary of foods that was originally intended to be just for transformations. This is the “replacementGuide” variable in recipeTransformer.py.
* If ConceptNet, the hard-coding, and the parser all failed, we just defaulted to the parser’s “root”, since the children of the root word always tend to have the other information that we are looking for.
