# FAQ

### Should I use a class or function for my tempo artifact?

Use a function if you have simple stateless inference logic. If you need to separate your inference code into multiple methods or need local state or specialized setup then use a class.


### For a class based Tempo artifact should I save the class or an instance of the class?

You should save the Class if you want the class `__init__` method to be called at runtime. If you save the instance then Tempo will attempt to pickle the state of the class instance which may cause issues for complex state objects such as Tensorflow graphs. If in doubt save the class.

