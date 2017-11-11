class A:
    @property
    def a(self):
         return 1

    def x(self):
        #return self.a
        return self.y()

    def y(self):
        return 1

class B(A):
    @property
    def a(self):
         return 2

    def x(self):
         # return super().x()
         return super(B, self).x()

    def y(self):
        return 2

print(A().x())
print(B().x())
