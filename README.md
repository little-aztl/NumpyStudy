# NumpyStudy
## 基础知识

`ndarray.ndim`数组的轴的个数

`ndarray.shape`数组的维度（一个整数的元组）

`ndarray.size`数组元素的总数。（shape的乘积）

`ndarray.dtype`数组中的元素类型

## 数组创建

`a = np.array([1, 2, 3, 4])`

`b = np.array([(1.5, 2.3), (4, 5, 6)])`

`c = np.array([[1, 2], [3, 4]], dtype=complex)`



+ `zeros`创建一个由0组成的数组，`ones`创建一个由1组成的数组，`empty`创建一个初始内容随机的数组。**默认情况下，数组的dtype是`float64`类型的**。

    `np.zeros((3, 4), dtype=np.int16)`

+ 类似于`range`的函数

    `np.arrange(10, 30, 5)`[10, 30)，步长为5

    `np.linspace(0, 2, 9)`9个数等距分布，第一个是0，最后一个是2

## 打印数组

当您打印数组时，NumPy以与嵌套列表类似的方式显示它，但具有以下布局：

- 最后一个轴从左到右打印，
- 倒数第二个从上到下打印，
- 其余部分也从上到下打印，每个切片用空行分隔。

然后将一维数组打印为行，将二维数据打印为矩阵，将三维数据打印为矩数组表。

```python
>>> c = np.arange(24).reshape(2,3,4)         # 3d array
>>> print(c)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

如果数组太大，Numpy会自动跳过数组的中心部分并只打印角点。要禁用此行为并强制Numpy打印整个数组，可以：`np.set_printoptions(threshold=sys.maxsize) `

## 基本操作

相同大小的矩阵可以进行加减乘除运算。

数学中的矩阵乘积可以使用`@`运算符或`dot`函数执行

`A @ B`或`A.dot(B)`

当使用不同类型的数组进行操作时，结果数组的类型对应于更一般或更精确的数组（int转float）

### 常用方法

```python
a.min()
a.max()
a.sum()
```

### 通过指定`axis`参数，可以沿数组的指定轴应用操作。

```python
>>> b = np.arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

### 通函数

sin，cos，exp等

```python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])
```

## 索引、切片和迭代

**一维**的数组可以进行索引、切片和迭代操作。

```python
a[2]
a[2:5]
a[:6:2]
a[::-1]
```

> Tips：`fromfunction`函数，可以得到一个数组，该数组的每个元素是将该元素的下标带入指定函数所得到的结果。
>
> ```python
> >>> def f(x,y):
> ...     return 10*x+y
> ...
> >>> b = np.fromfunction(f,(5,4),dtype=int)
> >>> b
> array([[ 0,  1,  2,  3],
>        [10, 11, 12, 13],
>        [20, 21, 22, 23],
>        [30, 31, 32, 33],
>        [40, 41, 42, 43]])
> ```

**多维**的数组每个轴可以有一个索引。

```python
b[2, 3]
b[0:5, 1]
b[:, 1]
b[1:3, :]
```

当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片。

如果想要对数组中的每个元素执行操作，可以使用`flat`属性。

```python
for element in b.flat:
    	print(element)
```

## 形状操纵

以下三个命令都返回一个修改后的数组，**而不会更改原数组**。

`a.ravel()`返回扁平后的数组

`a.reshape(6, 2)`第一轴有6维，第二轴有2维（六行两列）

> 如果将size指定为-1，则会自动计算。
>
> ```python
> >>> a.reshape(3,-1)
> array([[ 2.,  8.,  0.,  6.],
>        [ 4.,  5.,  1.,  1.],
>        [ 8.,  9.,  3.,  6.]])
> ```

`a.T`转置

---

`ndarray.resize`方法会修改数组本身。

`a.resize((2, 6))`

## 拷贝

### 完全不复制

简单的复制不会复制数组对象或其他数据。

```python
>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
```

**函数调用也是简单复制**，即传递的参数仍然是**引用**。

### 浅拷贝（视图）

`view`方法创建一个查看**相同数据**的新数组对象。

改A数组中的数据，B数组也会改；改A数组的大小，B数组的大小不会改。

```python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c.shape = 2,6                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```



**切片数组**本质就是返回一个视图

### 深拷贝

`copy`方法

```python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

