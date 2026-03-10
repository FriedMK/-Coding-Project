################### DO NOT MODIFIED THE CODE ###################
import numpy as np

class Initializer:

    def __call__(self, shape):
        return self.init(shape).astype(np.float32)  #将初始化的权重转换为32位浮点数，以节省内存并提高计算效率

    def init(self, shape):
        raise NotImplementedError


class XavierUniform(Initializer): #Xavier均匀分布初始化方法
    """
    Implement the Xavier method described in
    "Understanding the difficulty of training deep feedforward neural networks"
    Glorot, X. & Bengio, Y. (2010)
    Weights will have values sampled from uniform distribution U(-a, a) where
    a = gain * sqrt(6.0 / (num_in + num_out))
    """

    def __init__(self, gain=1.0):
        self._gain = gain

    def init(self, shape):
        fan_in, fan_out = self.get_fans(shape)
        a = self._gain * np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(low=-a, high=a, size=shape)

    def get_fans(self, shape):
        fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
        fan_out = shape[1] if len(shape) == 2 else shape[0]
        return fan_in, fan_out


class Constant(Initializer):

    def __init__(self, val):
        self._val = val

    def init(self, shape):
        return np.full(shape=shape, fill_value=self._val)


class Zeros(Constant):

    def __init__(self):
        super(Zeros, self).__init__(0.0)


class module():

    def __init__(self, **kwargs):  #**kwargs表示接受任意数量的关键字参数，并将它们作为一个字典传递给函数
        self.params = {p: None for p in self.param_names}
        self.ut_params = {p: None for p in self.ut_param_names}

        self.grads = {}
        self.shapes = {}

        self.training = True
        self.is_init = False

    def _forward(self, X, **kwargs):
        raise NotImplementedError

    def _backward(self, d, **kwargs):
        raise NotImplementedError

    def set_phase(self, phase): 
        self.training = phase.lower() == "train"

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self): #用于定义对象的字符串表示形式，常用于调试和打印日志
        shape = None if not self.shapes else self.shapes
        return "module: %s \t shape: %s" % (self.name, shape)

    @property
    def param_names(self):
        return ()

    def _init_params(self):
        for p in self.param_names:
            self.params[p] = self.initializers[p](self.shapes[p])
        self.is_init = True

    @property
    def ut_param_names(self):
        return ()


class Activation(module):  #激活函数模块，
                           #继承自moudle模块，定义所有激活函数需要一个func方法计算

    def __init__(self):
        super().__init__()
        self.inputs = None

    def _forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def func(self, x):
        raise NotImplementedError

def hello():
    print("Hello from your TAs!")
