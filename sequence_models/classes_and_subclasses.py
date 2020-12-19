class MyClass:
    def __init__(self, y):
        self.x = y

    def __call__(self, z):
        self.x += z
        print(self.x)

instance_d = MyClass(5)
instance_d(10)


class MyClass:

    def __init__(self, y, z):
        self.x_1 = y
        self.x_2 = z

    def __call__(self):
        result = self.x_1 + self.x_2

        print(f"Addition of {self.x_1} and {self.x_2} is {result}")
        return result


instance_e = MyClass(10, 15)

def test_class_definition():

    assert instance_e.x_1 == 10, "check the value assigned to x_1"
    assert instance_e.x_2 == 15, "check the value assigned to x_2"
    assert instance_e() == 25, "check the __call__ method"

    print("\33[92mall tests are passed")


test_class_definition()


class MyClass:

    def __init__(self, y, z):
        self.x_1 = y
        self.x_2 = z

    def __call__(self):
        a = self.x_1 + self.x_2
        return a

    def my_method(self, w):
        b = self.x_1*self.x_2 + w
        return b

    def new_method(self, v):
        result = self.my_method(v)
        return result


instance_g = MyClass(1, 10)

print(f"output of my_method: {instance_g.my_method(16)}")
print(f"output of new_method: {instance_g.new_method(16)}")



'''
Inheritance
'''

class sub_c(MyClass): # subclass sub_c from MyClass
    def my_method(self):
        b = self.x_1*self.x_2
        return b

test = sub_c(3, 10)
assert test.my_method() == 30, "the method should return the product between x_1 and x_2"

print(f"output of overriddenn my_method of test: {test.my_method()}")