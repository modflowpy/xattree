# Tutorial

## Class decorator

## Field decorators

### Dim

### Coord

### Array

### Child

#### Collections

Child collections give you more flexibility in naming your children. Dictionaries and lists are supported. `xattree` will "flatten" dictionaries and lists of child types before attaching them to the `DataTree`. Changes applied to the attribute collection via builtin methods will be reflected in the data tree's children and vice versa.

Use a dictionary of child types if you want to name them not up front but at their moment of birth, which is understandable. 

Using a list is effectively to declare that you don't want to have to name them, which, while deplorable among our kind, is standard feline conduct, therefore `xattree` accepts anonymous children. But it will name them behind your back, appending an auto-incrementing integer to the name of the list field.

While `xattree` will raise an error at runtime if a user-specified or auto-generated name collides with another fields, it's best to name fields such that collisions are impossible.
