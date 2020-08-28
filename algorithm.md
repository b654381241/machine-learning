# 机器学习Algorithm

## 基础

##### 常用的核函数：

1. Linear Kernel
   线性核是最简单的核函数，核函数的数学公式如下：

![img](algorithm.assets/1598911-20190417152132247-1752637987.png)

如果我们将线性核函数应用在KPCA中，我们会发现，推导之后和原始PCA算法一模一样，很多童鞋借此说“kernel is shit！！！”，这是不对的，这只是线性核函数偶尔会出现等价的形式罢了。

2. Polynomial Kernel
   多项式核实一种非标准核函数，它非常适合于正交归一化后的数据，其具体形式如下：

![img](algorithm.assets/1598911-20190417152240458-973995769.png)

这个核函数是比较好用的，就是参数比较多，但是还算稳定。

3. Gaussian Kernel
   这里说一种经典的鲁棒径向基核，即高斯核函数，鲁棒径向基核对于数据中的噪音有着较好的抗干扰能力，其参数决定了函数作用范围，超过了这个范围，数据的作用就“基本消失”。高斯核函数是这一族核函数的优秀代表，也是必须尝试的核函数，其数学形式如下：

![img](algorithm.assets/1598911-20190417152155489-953622202.png)
虽然被广泛使用，但是这个核函数的性能对参数十分敏感，以至于有一大把的文献专门对这种核函数展开研究，同样，高斯核函数也有了很多的变种，如指数核，拉普拉斯核等。

4. Exponential Kernel
   指数核函数就是高斯核函数的变种，它仅仅是将向量之间的L2距离调整为L1距离，这样改动会对参数的依赖性降低，但是适用范围相对狭窄。其数学形式如下：

![img](algorithm.assets/1598911-20190417152214893-1005745352.png)

5. Laplacian Kernel
   拉普拉斯核完全等价于指数核，唯一的区别在于前者对参数的敏感性降低，也是一种径向基核函数。

![img](algorithm.assets/1598911-20190417152259072-2139358069.png)

6. ANOVA Kernel
   ANOVA 核也属于径向基核函数一族，其适用于多维回归问题，数学形式如下：

![img](algorithm.assets/1598911-20190417152312214-622763030.png)

7. Sigmoid Kernel
   Sigmoid 核来源于神经网络，现在已经大量应用于深度学习，是当今机器学习的宠儿，它是S型的，所以被用作于“激活函数”。关于这个函数的性质可以说好几篇文献，大家可以随便找一篇深度学习的文章看看。

![img](algorithm.assets/1598911-20190417152323856-1485057688.png)

8. Rational Quadratic Kernel
   二次有理核完完全全是作为高斯核的替代品出现，如果你觉得高斯核函数很耗时，那么不妨尝试一下这个核函数，顺便说一下，这个核函数作用域虽广，但是对参数十分敏感，慎用！！！！

![img](algorithm.assets/1598911-20190417152336227-1386121825.png)

9. Multiquadric Kernel
   多元二次核可以替代二次有理核，它是一种非正定核函数。

![img](algorithm.assets/1598911-20190417152349673-1989203350.png)

10. Inverse Multiquadric Kernel
    顾名思义，逆多元二次核来源于多元二次核，这个核函数我没有用过，但是据说这个基于这个核函数的算法，不会遇到核相关矩阵奇异的情况。

![img](algorithm.assets/1598911-20190417152403015-1338036097.png)

 

11. Circular Kernel
    这个核函数没有用过，其数学形式如下所示：

![img](algorithm.assets/1598911-20190417152413475-1864044724.png)

 

12. Spherical Kernel
    这个核函数是上一个的简化版，形式如下所示

![img](algorithm.assets/1598911-20190417152423071-1519802269.png)

13. Wave Kernel
    这个核函数没有用过，其适用于语音处理场景。

![img](algorithm.assets/1598911-20190417152432417-1633392119.png)

14. Triangular Kernel
    三角核函数感觉就是多元二次核的特例，数学公式如下：

![img](algorithm.assets/1598911-20190417152443837-707071513.png)

15. Log Kernel
    对数核一般在图像分割上经常被使用，数学形式如下：

![img](algorithm.assets/1598911-20190417152508120-1917221316.png)

16. Spline Kernel

![img](algorithm.assets/1598911-20190417152519701-1326338724.png)

17. Bessel Kernel

![img](algorithm.assets/1598911-20190417152531062-127498552.png)

18. Cauchy Kernel
    柯西核来源于神奇的柯西分布，与柯西分布相似，函数曲线上有一个长长的尾巴，说明这个核函数的定义域很广泛，言外之意，其可应用于原始维度很高的数据上。

![img](algorithm.assets/1598911-20190417152541590-877150296.png)

19. Chi-Square Kernel
    卡方核，这是我最近在使用的核函数，让我欲哭无泪，在多个数据集上都没有用，竟然比原始算法还要差劲，不知道为什么文献作者首推这个核函数，其来源于卡方分布，数学形式如下：

![img](algorithm.assets/1598911-20190417152554522-1695248644.png)

它存在着如下变种：

![img](algorithm.assets/1598911-20190417152604823-1101248110.png)

其实就是上式减去一项得到的产物，这个核函数基于的特征不能够带有赋值，否则性能会急剧下降，如果特征有负数，那么就用下面这个形式：

![img](algorithm.assets/1598911-20190417152619708-574538469.png)

20. Histogram Intersection Kernel
    直方图交叉核在图像分类里面经常用到，比如说人脸识别，适用于图像的直方图特征，例如extended LBP特征其数学形式如下，形式非常的简单

![img](algorithm.assets/1598911-20190417152629659-266306348.png)

21. Generalized Histogram Intersection
    顾名思义，广义直方图交叉核就是上述核函数的拓展，形式如下：

![img](algorithm.assets/1598911-20190417152640374-1791429122.png)

22. Generalized T-Student Kernel
    TS核属于mercer核，其数学形式如下，这个核也是经常被使用的

![img](algorithm.assets/1598911-20190417152650020-1727166888.png)

### 激活函数

#### 1. Sigmoid激活函数

- 函数表达式：

​      ![img](algorithm.assets/1583235-20200226152008245-782955790.png)

- 函数图像：

![img](algorithm.assets/1583235-20200226152112616-2051833875.png)





![img](algorithm.assets/1583235-20200226152317949-1705481847.png)

- 优点：Sigmoid激活函数是应用范围最广的一类激活函数，具有指数形状，它在物理意义上最为接近生物神经元。另外，Sigmoid的输出是`(0,1)`，具有很好的性质，可以被表示为概率或者用于输入的归一化等。可以看出，Sigmoid函数连续，光滑，严格单调，以`(0,0.5)`中心对称，是一个非常良好的阈值函数。当`x`趋近负无穷时，`y`趋近于`0`；`x`趋近于正无穷时，`y`趋近于`1`；`x=0`时，`y=0.5`。当然，在`x`超出`[-6,6]`的范围后，函数值基本上没有变化，值非常接近，在应用中一般不考虑。Sigmoid函数的导数是其本身的函数，即`f′(x)=f(x)(1−f(x))`，计算非常方便，也非常节省计算时间。
- 缺点：Sigmoid最明显的缺点就是饱和性。从曲线图中看到，其两侧的导数逐渐趋近于`0`，即：limx->∞f'(x)=0 。我们将具有这种性质的激活函数叫作软饱和激活函数。具体的，饱和又可分为左饱和与右饱和。与软饱和对应的是硬饱和, 即`f′(x)=0`，当`|x|>c`，其中`c`为常数。sigmoid 的软饱和性，使得深度神经网络在二三十年里一直难以有效的训练，是阻碍神经网络发展的重要原因。另外，Sigmoid函数的输出均大于`0`，使得输出不是`0`均值，这称为偏移现象，这会导致后一层的神经元将得到上一层输出的非`0`均值的信号作为输入。

#### 2. TanH

- 函数表达式：

![img](algorithm.assets/1583235-20200226152435146-209407762.png)

 

-  *函数图像：*

![img](algorithm.assets/1583235-20200226152509013-1311030532.png)

 

- 导数：

![img](algorithm.assets/1583235-20200226152609444-17639619.png)

 

 

 

- 优点：与Sigmoid相比，它的输出均值是`0`，使得其收敛速度要比Sigmoid快，减少迭代次数。
- 缺点：该导数在正负饱和区的梯度都会接近于`0`值(仍然具有软饱和性)，会造成梯度消失。还有其更复杂的幂运算。

#### 3. ReLU

- 函数表达式：

![img](algorithm.assets/1583235-20200226152654990-637907913.png) 

- *函数图像：*

![img](algorithm.assets/1583235-20200226152827598-124789017.png)

 

 

- 导数：当 x>0 时， f'(x)=1 ,当 x<0 时 f'(x)=0 。
- 优点：ReLU的全称是Rectified Linear Units，是一种AlexNet时期才出现的激活函数。可以看到，当`x<0`时，ReLU硬饱和，而当`x>0`时，则不存在饱和问题。所以，ReLU 能够在`x>0`时保持梯度不衰减，从而缓解梯度消失问题。这让我们能够直接以监督的方式训练深度神经网络，而无需依赖无监督的逐层预训练。
- 缺点：随着训练的推进，部分输入会落入硬饱和区，导致对应权重无法更新。这种现象被称为“神经元死亡”。与Sigmoid类似，ReLU的输出均值也大于`0`，偏移现象和神经元死亡会共同影响网络的收敛性。

#### 4. Leaky ReLU & PReLU

- 函数表达式和导数：

![img](algorithm.assets/1583235-20200226153147181-4279586.png)

 

 

- 函数图像：

![img](algorithm.assets/1583235-20200226153222935-1691580385.png)

- 特点：为了改善ReLU在  x<0  时梯度为  0造成Dead ReLU，提出了Leaky ReLU使得这一问题得到了缓解。例如在我们耳熟能详的YOLOV3网络中就使用了Leaky ReLU这一激活函数，一般 α取  0.25。另外PReLU就是将Leaky ReLU公式里面的 α当成可学习参数参与到网络训练中。

#### 5. ReLU6

- 函数表达式：

![img](algorithm.assets/1583235-20200226153851366-2142643213.png)

- 特点：ReLU6就是普通的ReLU但是限制最大输出值为`6`（对输出值做`clip`），这是为了在移动端设备`float16`的低精度的时候，也能有很好的数值分辨率，如果对ReLU的激活范围不加限制，输出范围为 0到正无穷，如果激活值非常大，分布在一个很大的范围内，则低精度的`float16`无法很好地精确描述如此大范围的数值，带来精度损失。

#### 6. ELU

- 函数表达式：

![img](algorithm.assets/1583235-20200226153945254-1049604681.png)

 

 

- 函数图像：

![img](algorithm.assets/1583235-20200226154315024-1071537090.png)

- 导数：当 x>0 时， f'(x)=1 ，当 x<0 时， f'(x)=αex 。
- 特点：融合了sigmoid和ReLU，左侧具有软饱和性，右侧无饱和性。右侧线性部分使得ELU能够缓解梯度消失，而左侧软饱能够让ELU对输入变化或噪声更鲁棒。ELU的输出均值接近于零，所以收敛速度更快。在 ImageNet上，不加Batch Normalization 30层以上的ReLU网络会无法收敛，PReLU网络在MSRA的Fan-in （caffe ）初始化下会发散，而 ELU 网络在Fan-in/Fan-out下都能收敛。

#### 7. SoftSign

- 函数表达式：

![img](algorithm.assets/1583235-20200226154533757-1795270793.png)

- 函数图像：

![img](algorithm.assets/1583235-20200226154601500-2071763039.png)

 

 

- 导数：图中已经求出。
- 特点：Softsign是tanh激活函数的另一个替代选择，从图中可以看到它和tanh的曲线极其相似，不过相比于tanh，**Softsign的曲线更平坦，导数下降的更慢一点，这个特性使得它可以缓解梯度消失问题，可以更高效的学习。**

#### 8. SoftPlus

- 函数表达式：

![img](algorithm.assets/1583235-20200226154726496-1222801327.png)

 

 

- 函数图像：

![img](algorithm.assets/1583235-20200226154747950-1887446719.png)

 

 

- 函数导数：SoftPlus激活函数的导数恰好就是sigmoid激活函数，即 f'(x)=sigmoid(x) 。
- 优点：**SoftPlus可以作为ReLu的一个不错的替代选择，可以看到与ReLU不同的是，SoftPlus的导数是连续的、非零的、无处不在的，这一特性可以防止出现ReLU中的“神经元死亡”现象。**
- 缺点：**SoftPlus是不对称的，不以0为中心，存在偏移现象；而且，由于其导数常常小于1，也可能会出现梯度消失的问题。**

#### 9. SELU

- 函数表达式： SELU(x)=λ*ELU(x) ，也即是：

![img](algorithm.assets/1583235-20200226154955056-1281408154.png)

- 特点：这个激活函数来自论文：https://arxiv.org/abs/1706.02515 。而这篇论文就是提出了这一激活函数，然后论文写了93页公式来证明**只需要把激活函数换成SELU就能使得输入在经过一定层数之后变成固定的分布**。。而这个函数实际上就是在ELU激活函数的基础上乘以了一个 λ ，但需要注意的是这个 λ是大于 1的。

#### 10. Swish

- 函数表达式： f(x) = x*sigmoid(x) ，其中 β 是个常数或可训练的参数.Swish 具备无上界有下界、平滑、非单调的特性。
- 函数图像：

![img](algorithm.assets/1583235-20200226155211653-390652468.png)

 

- 函数导数

![img](algorithm.assets/1583235-20200226155259119-1900887328.png)

- 特点：**Swish 在深层模型上的效果优于 ReLU**。例如，仅仅使用 Swish 单元替换 ReLU 就能把 **Mobile NASNetA** 在 ImageNet 上的 top-1 分类准确率提高  0.9% ，**Inception-ResNet-v**的分类准确率提高  0.6% 。当 β=0 时，Swish激活函数变成线性函数 f(x)=x/2 .而当 β->∞ 时， δ（x）=(1+exp(-x))-1 为0或1,这个时候Swish激活函数变成ReLU激活函数 f(x)=2max(0,x) 。因此Swish激活函数可以看做是介于线性函数与ReLU函数之间的平滑函数。

#### 11. Maxout

- 函数表达式：

![img](algorithm.assets/1583235-20200226155843327-1708078303.png)

 

 

- *特点： Maxout 模型实际上也是一种新型的激活函数，在前馈式神经网络中， Maxout 的输出即取该层的最大值，在卷积神经网络中，一个 Maxout 特征图可以是由***多个特征图取最值得到***。 Maxout 的拟合能力是非常强的，它可以拟合任意的的凸函数。但是它又和* Dropout 一样需要人为设定一个 k 值。为了便于理解，假设有一个在第 i 层有 2 个节点， i+1 层有1个节点构成的神经网络。即：

![img](algorithm.assets/1583235-20200226160009332-218219140.png)

 

 

- 激活值 out = f(W*X+b) ，其中 f是激活函数， *在这里代表內积。然后 X=(x1,x2)T ， W=(w1,w2)T 。那么当我们对 i层使用 Maxout （设定 k=5 ）然后再输出的时候，情况就发生了改变。网络就变成了：

![img](algorithm.assets/1583235-20200226160403571-1008757157.png)

 

-  此时网络形式上就变成上面的样子，用公式表现出来就是： z1=W1*X+b1 ， z2=W2*X+b2 ， z3=W3*X+b3 ， z4=W4*X+b4 ， z5=W5*X+b5 。 out=max(z1,z2,z3,z4,z5) 也就是说第层的激活值计算了5次，可我们明明只需要 1个激活值，那么我们该怎么办？其实上面的叙述中已经给出了答案，取这 5个的最大值来作为最终的结果。
- 可以看到采用 Maxout 的话参数个数也增加了 k倍，计算开销会增大。

#### 12. Mish

- 函数表达式：

![img](algorithm.assets/1583235-20200226194406278-869249982.png)

 

 

 

- 函数图像：

![img](algorithm.assets/1583235-20200226194423910-376614584.png)

- 特点：这个激活函数是最新的SOTA激活函数。论文中提到，以上无边界(即正值可以达到任何高度)避免了由于封顶而导致的饱和，理论上对负值的轻微允许更好的梯度流，而不是像ReLU中那样的硬零边界，并且整个损失函数仍然保持了平滑性。

## 聚类

### kmeans

K-Means算法是一种无监督分类算法，假设有无标签数据集：

![X= \left[ \begin{matrix} x^{(1)} \\ x^{(2)} \\ \vdots \\ x^{(m)} \\ \end{matrix} \right]](https://math.jianshu.com/math?formula=X%3D%20%5Cleft%5B%20%5Cbegin%7Bmatrix%7D%20x%5E%7B(1)%7D%20%5C%5C%20x%5E%7B(2)%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20x%5E%7B(m)%7D%20%5C%5C%20%5Cend%7Bmatrix%7D%20%5Cright%5D)

该算法的任务是将数据集聚类成![k](https://math.jianshu.com/math?formula=k)个簇![C={C_{1},C_{2},...,C_{k}}](https://math.jianshu.com/math?formula=C%3D%7BC_%7B1%7D%2CC_%7B2%7D%2C...%2CC_%7Bk%7D%7D)，最小化损失函数为：

![E=\sum_{i=1}^{k}\sum_{x\in{C_{i}}}||x-\mu_{i}||^{2}](https://math.jianshu.com/math?formula=E%3D%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%5Csum_%7Bx%5Cin%7BC_%7Bi%7D%7D%7D%7C%7Cx-%5Cmu_%7Bi%7D%7C%7C%5E%7B2%7D)

其中![\mu_{i}](https://math.jianshu.com/math?formula=%5Cmu_%7Bi%7D)为簇![C_{i}](https://math.jianshu.com/math?formula=C_%7Bi%7D)的中心点：

![\mu_{i}=\frac{1}{|C_{i}|}\sum_{x\in{C{i}}}x](https://math.jianshu.com/math?formula=%5Cmu_%7Bi%7D%3D%5Cfrac%7B1%7D%7B%7CC_%7Bi%7D%7C%7D%5Csum_%7Bx%5Cin%7BC%7Bi%7D%7D%7Dx)

要找到以上问题的最优解需要遍历所有可能的簇划分，K-Mmeans算法使用贪心策略求得一个近似解，具体步骤如下：

1. 在样本中随机选取![k](https://math.jianshu.com/math?formula=k)个样本点充当各个簇的中心点![\{\mu_{1},\mu_{2},...,\mu_{k}\}](https://math.jianshu.com/math?formula=%5C%7B%5Cmu_%7B1%7D%2C%5Cmu_%7B2%7D%2C...%2C%5Cmu_%7Bk%7D%5C%7D) 

2. 计算所有样本点与各个簇中心之间的距离![dist(x^{(i)},\mu_{j})](https://math.jianshu.com/math?formula=dist(x%5E%7B(i)%7D%2C%5Cmu_%7Bj%7D))，然后把样本点划入最近的簇中![x^{(i)}\in{\mu_{nearest}}](https://math.jianshu.com/math?formula=x%5E%7B(i)%7D%5Cin%7B%5Cmu_%7Bnearest%7D%7D) 

3. 根据簇中已有的样本点，重新计算簇中心
    ![\mu_{i}:=\frac{1}{|C_{i}|}\sum_{x\in{C{i}}}x](https://math.jianshu.com/math?formula=%5Cmu_%7Bi%7D%3A%3D%5Cfrac%7B1%7D%7B%7CC_%7Bi%7D%7C%7D%5Csum_%7Bx%5Cin%7BC%7Bi%7D%7D%7Dx) 

4. 重复2、3

### **改进一——kmean++**

![img](https://images2018.cnblogs.com/blog/1366181/201804/1366181-20180402200209017-1976662980.png)

### **改进二——Kernel K-means**

​    设数据集![img](http://img.blog.csdn.net/20140616121824875)，其中![img](http://img.blog.csdn.net/20140616121842109)，![img](http://img.blog.csdn.net/20140616121849937)。Mercer核函数![img](http://img.blog.csdn.net/20140616122357328)，根据Mercer定理存在映射![img](http://img.blog.csdn.net/20140616122431781)，使得![img](http://img.blog.csdn.net/20140616122453234)。

​        核K-均值聚类就是讨论映射数据集![img](http://img.blog.csdn.net/20140616122556390)在![img](http://img.blog.csdn.net/20140616122708109)空间中的聚类情况，设在![img](http://img.blog.csdn.net/20140616122708109)空间中，把数据集分为![img](http://img.blog.csdn.net/20140616122826578)类，![img](http://img.blog.csdn.net/20140616122922125)为第![img](http://img.blog.csdn.net/20140616122826578)类的均值，![img](http://img.blog.csdn.net/20140616123005000)。

即考虑以下模型：

![img](http://img.blog.csdn.net/20140616123528984)

![img](http://img.blog.csdn.net/20140616123542984)。



**问题1：**

怎么训练上述模型，因为![img](http://img.blog.csdn.net/20140616123740078)一般情况下是解不出来的。

方法：

初始化![img](http://img.blog.csdn.net/20140616123859968)，![img](http://img.blog.csdn.net/20140616123911593)，![img](http://img.blog.csdn.net/20140616123005000)，其中![img](http://img.blog.csdn.net/20140616133046875)，令

![img](http://img.blog.csdn.net/20140616133250859)，![img](http://img.blog.csdn.net/20140616123005000)。

**E步**：求![img](http://img.blog.csdn.net/20140616140421921)，

![img](http://img.blog.csdn.net/20140616140449843)

注意其中：

![img](http://img.blog.csdn.net/20140616142438250)，![img](http://img.blog.csdn.net/20140616142529593)。

**M步：**固定![img](http://img.blog.csdn.net/20140616140421921)，求![img](http://img.blog.csdn.net/20140616142958078)。

![img](http://img.blog.csdn.net/20140616143531703)，

![img](http://img.blog.csdn.net/20140616143837296),

![img](http://img.blog.csdn.net/20140616143848421)，

其中**，![img](http://img.blog.csdn.net/20140616123005000)。**

进入下一轮迭代，直至收敛！

### 改进三——ISODATA算法
​      **[1] 预期的聚类中心数目Ko**：虽然在ISODATA运行过程中聚类中心数目是可变的，但还是需要由用户指定一个参考标准。事实上，该算法的聚类中心数目变动范围也由**Ko**决定。具体地，最终输出的聚类中心数目范围是 [**Ko/2**, ***2Ko***]。

​       **[2] 每个类所要求的最少样本数目Nmin**：用于判断当某个类别所包含样本分散程度较大时是否可以进行分裂操作。如果分裂后会导致某个子类别所包含样本数目小于***Nmin***，就不会对该类别进行分裂操作。

​      **[3] 最大方差Sigma**：用于衡量某个类别中样本的分散程度。当样本的分散程度超过这个值时，则有可能进行分裂操作（注意同时需要满足**[2]**中所述的条件）。

​      **[4] 两个类别对应聚类中心之间所允许最小距离dmin**：如果两个类别靠得非常近（即这两个类别对应聚类中心之间的距离非常小），则需要对这两个类别进行合并操作。是否进行合并的阈值就是由***dmin***决定。

​      相信很多人看完上述输入的介绍后对ISODATA算法的流程已经有所猜测了。的确，ISODATA算法的原理非常直观，不过由于它和其他两个方法相比需要额外指定较多的参数，并且某些参数同样很难准确指定出一个较合理的值，因此ISODATA算法在实际过程中并没有K-means++受欢迎。

​      首先给出ISODATA算法主体部分的描述，如下图所示：

[![图4](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025949447-680611657.png)](http://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025947447-390971451.png)

**图4. ISODATA算法的主体部分**

​     上面描述中没有说明清楚的是第5步中的分裂操作和第6步中的合并操作。下面首先介绍合并操作：

[![图5](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025951775-1194408309.png)](http://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025950760-1467458924.png)

**图5. ISODATA算法的合并操作**

​     最后是ISODATA算法中的分裂操作。

[![图6](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025954494-895315300.png)](http://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025953056-677584793.png)

**图6. ISODATA算法的分裂操作**

​      最后，针对ISODATA算法总结一下：**该算法能够在聚类过程中根据各个类所包含样本的实际情况动态调整聚类中心的数目。如果某个类中样本分散程度较大（通过方差进行衡量）并且样本数量较大，则对其进行分裂操作；如果某两个类别靠得比较近（通过聚类中心的距离衡量），则对它们进行合并操作。**

​       可能没有表述清楚的地方是ISODATA-分裂操作的第1步和第2步。同样地以图三所示数据集为例，假设最初1，2，3，4，5，6，8号被分到了同一个类中，执行第1步和第2步结果如下所示：

[![图7](https://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025955213-1053926739.png)](http://images2015.cnblogs.com/blog/1024143/201701/1024143-20170111025954853-32277500.png)

​      而在正确分类情况下（即1，2，3，4为一类；5，6，7，8为一类），方差为0.33。因此，目前的方差远大于理想的方差，ISODATA算法就很有可能对其进行分裂操作。

### 改进对比

  (1) **K-means与K-means++：**原始K-means算法最开始随机选取数据集中K个点作为聚类中心，而K-means++按照如下的思想选取K个聚类中心：假设已经选取了n个初始聚类中心(0<n<K)，则在选取第n+1个聚类中心时：距离当前n个聚类中心越远的点会有更高的概率被选为第n+1个聚类中心。在选取第一个聚类中心(n=1)时同样通过随机的方法。可以说这也符合我们的直觉：聚类中心当然是互相离得越远越好。这个改进虽然直观简单，但是却非常得有效。

​      (2) **K-means与ISODATA：**ISODATA的全称是迭代自组织数据分析法。在K-means中，K的值需要预先人为地确定，并且在整个算法过程中无法更改。而当遇到高维度、海量的数据集时，人们往往很难准确地估计出K的大小。ISODATA就是针对这个问题进行了改进，它的思想也很直观：当属于某个类别的样本数过少时把这个类别去除，当属于某个类别的样本数过多、分散程度较大时把这个类别分为两个子类别。

​      (3) **K-means与Kernel K-means：**传统K-means采用欧式距离进行样本间的相似度度量，显然并不是所有的数据集都适用于这种度量方式。参照支持向量机中核函数的思想，将所有样本映射到另外一个特征空间中再进行聚类，就有可能改善聚类效果。

### **Mean-Shift 聚类**

1. 为了解释 mean-shift，我们将考虑一个二维空间中的点集，像上图所示那样。我们以一个圆心在C点（随机选择）的圆形滑窗开始，以半径 r 作为核。Mean shift 是一个爬山算法，它每一步都迭代地把核移动到更高密度的区域，直到收敛位置。
2. 在每次迭代时，通过移动中心点到滑窗中点的均值处，将滑窗移动到密度更高的区域（这也是这种算法名字的由来）。滑窗内的密度与在其内部点的数量成正比。很自然地，通过将中心移动到窗内点的均值处，可以逐步的移向有个高的密度的区域。
3. 我们继续根据均值来移动滑窗，直到有没有哪个方向可以使核中容纳更多的点。查看上面的图，我们一直移动圆圈直到密度不再增长。（即窗内点的数量不再增长）。
4. 用很多滑窗重复1-3这个过程，直到所有的点都包含在了窗内。当多个滑动窗口重叠时，包含最多点的窗口将被保留。然后，根据数据点所在的滑动窗口对数据点进行聚类。

下图展示了所有滑动窗口从端到端的整个过程。每个黑色的点都代表滑窗的质心，每个灰色的点都是数据点。

![img](https://hiphotos.baidu.com/feed/pic/item/f7246b600c3387447066b8b65c0fd9f9d62aa05d.jpg)

Mean-Shift 聚类的全部过程

与 K-means 聚类不同的是，Mean-Shift 不需要选择聚类的数量，因为mean-shift 自动发现它。这是一个很大的优点。事实上聚类中心向着有最大密度的点收敛也是我们非常想要的，因为这很容易理解并且很适合于自然的数据驱动的场景。缺点是滑窗尺寸/半径“r“的选择需要仔细考虑。

### DBSCAN

1. DBSCAN 从一个任意的还没有被访问过的启动数据点开始。用一个距离 epsilon ε 将这个点的邻域提取出来（所有再距离 ε 内的点都视为邻居点）。

2. 如果在邻域内有足够数量的点（根据 minPoints) ，那么聚类过程开始，并且当前数据点变成新集群中的第一个点。否则，该点将被标记为噪声（之后这个噪声点可能会变成集群中的一部分）。在这两种情况中的点都被标记为”已访问“。

3. 对于这个新集群中的第一个点，在它 ε 距离邻域内的点已将变成相同集群中的一部分。这个让所有在 ε 邻域内的点都属于相同集群的过程在之后会一直被重复做，直到所有新点都被加进集群分组中。

4. 第 2，3 步的过程会一直重复直到集群内所有点都被确定，即所有在 ε 邻域内的点都被访问且被打上标签。

5. 一旦我们在当前集群做完这些，一个新的未被访问的点会被提取并处理，从而会接着发现下一个集群或噪声。这个过程反复进行直到所有的点都被编辑为已访问。既然在最后所有的点都被访问，那么每个点都被标记为属于一个集群或者是噪声。

#### 优缺点

相较于其他聚类算法，DBSCAN 提出了一些很棒的优点。首先，它根本不需要预置集群的数量。它还将离群值认定为噪声，不像 mean-shift 中仅仅是将它们扔到一个集群里，甚至即使该数据点的差异性很大也这么做。另外，这个算法还可以很好的找到任意尺寸核任意形状的集群。

DBSCAN 最大的缺点是当集群的密度变化时，它表现的不像其他算法那样好。这是因为当密度变化时，距离的阈值 ε 和用于确定邻居点的 minPoints 也将会随之改变。这个缺点也会发生在很高为的数据中，因为距离阈值 ε 变得很难被估计。

### GMM

####一、GMM概述

![img](https://img2018.cnblogs.com/blog/1027447/201809/1027447-20180914192745456-2065802777.png)



#### 二、GMM算法步骤

![img](https://img2018.cnblogs.com/blog/1027447/201809/1027447-20180914192859284-1014242477.png)

#### 三、总结

1. GMM算法中间参数估计部分用到了EM算法，EM算法分为两步：

​      （1）E步：求目标函数期望，更多的是求目标函数取对数之后的期望值。

​      （2）M步：使期望最大化。用到极大似然估计，拉格朗日乘数法，对参数求偏导，最终确定新的参数。

2. K-means，FCM与GMM算法参数估计的数学推导思路大体一致，都先确立目标函数，然后使目标函数最大化的参数取值就是迭代公式。

3. 三个算法都需要事先指定k。K-means与FCM中的k指的是要聚的类的个数，GMM算法中的k指的是k个单高斯混合模型。

4. 三个算法流程一致：

​    （1）通过一定的方法初始化参数（eg:随机，均值······）

​    （2）确立目标函数

​    （3）通过一定的方法使目标函数最大化，更新参数迭代公式（eg:EM，粒子群······）

​    （4）设置一定的终止条件，使算法终止。若不满足条件，转向（3）

### 层次聚类

层次聚类算法分为两类：自上而下和自下而上。凝聚层级聚类(HAC)是自下而上的一种聚类算法。HAC首先将每个数据点视为一个单一的簇，然后计算所有簇之间的距离来合并簇，知道所有的簇聚合成为一个簇为止。
下图为凝聚层级聚类的一个实例：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180301171047257?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS2F0aGVyaW5lX2hzcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

具体步骤：

1. 首先我们将每个数据点视为一个单一的簇，然后选择一个测量两个簇之间距离的度量标准。例如我们使用average linkage作为标准，它将两个簇之间的距离定义为第一个簇中的数据点与第二个簇中的数据点之间的平均距离。

2. 在每次迭代中，我们将两个具有最小average linkage的簇合并成为一个簇。

3. 重复步骤2知道所有的数据点合并成一个簇，然后选择我们需要多少个簇。

层次聚类优点：（1）不需要知道有多少个簇 （2）对于距离度量标准的选择并不敏感



### 谱聚类

#### 1.1 谱和谱聚类

##### 1.1.1 谱

方阵作为线性算子，它的所有特征值的全体统称为方阵的谱。方阵的谱半径为最大的特征值。矩阵A的谱半径是矩阵![A^TA](algorithm.assets/gif-1598506970556.gif)的最大特征值。

##### 1.1.2 谱聚类

谱聚类是一种基于图论的聚类方法，通过对样本数据的拉普拉斯矩阵的特征向量进行聚类，从而达到对样本数据聚类的母的。谱聚类可以理解为将高维空间的数据映射到低维，然后在低维空间用其它聚类算法（如KMeans）进行聚类。

##### 1.2 谱聚类算法简单描述

输入：n个样本点![X=\left \{ x{_{1}},x{_{2}},...,x{_{n}} \right \}](https://private.codecogs.com/gif.latex?X%3D%5Cleft%20%5C%7B%20x%7B_%7B1%7D%7D%2Cx%7B_%7B2%7D%7D%2C...%2Cx%7B_%7Bn%7D%7D%20%5Cright%20%5C%7D)和聚类簇的数目k；

输出：聚类簇![A{_{1}},A{_{2}},...,A{_{k}}](algorithm.assets/gif.gif)

（1）使用下面公式计算![n*n](algorithm.assets/gif-1598506969103.gif)的相似度矩阵W；

​                                 ![s{_{ij}}=s(x{_{i}},x{_{j}})=\sum_{i=1,j=1}^{n}exp\frac{-||x{_{i}}-x{_{j}}||^2}{2\sigma ^2}](https://private.codecogs.com/gif.latex?s%7B_%7Bij%7D%7D%3Ds%28x%7B_%7Bi%7D%7D%2Cx%7B_%7Bj%7D%7D%29%3D%5Csum_%7Bi%3D1%2Cj%3D1%7D%5E%7Bn%7Dexp%5Cfrac%7B-%7C%7Cx%7B_%7Bi%7D%7D-x%7B_%7Bj%7D%7D%7C%7C%5E2%7D%7B2%5Csigma%20%5E2%7D)

W为![s{_{ij}}](algorithm.assets/gif-1598506969295.gif)组成的相似度矩阵。

（2）使用下面公式计算度矩阵D；

​                                ![d{_{i}}=\sum_{j=1}^{n}w{_{ij}}](algorithm.assets/gif-1598506969358.gif)，即相似度矩阵W的每一行元素之和

D为![d{_{i}}](algorithm.assets/gif-1598506969455.gif)组成的![n*n](algorithm.assets/gif-1598506969103.gif)对角矩阵。

（3）计算拉普拉斯矩阵![L=D-W](algorithm.assets/gif-1598506969512.gif)；

（4）计算L的特征值，将特征值从小到大排序，取前k个特征值，并计算前k个特征值的特征向量![u{_{1}},u{_{2}},...,u{_{k}}](algorithm.assets/gif-1598506969559.gif)；

（5）将上面的k个列向量组成矩阵![U=\left \{ u{_{1}},u{_{2}},...,u{_{k}} \right \}](https://private.codecogs.com/gif.latex?U%3D%5Cleft%20%5C%7B%20u%7B_%7B1%7D%7D%2Cu%7B_%7B2%7D%7D%2C...%2Cu%7B_%7Bk%7D%7D%20%5Cright%20%5C%7D)，![U\in R^{n*k}](https://private.codecogs.com/gif.latex?U%5Cin%20R%5E%7Bn*k%7D)；

（6）令![y{_{i}}\in R^k](https://private.codecogs.com/gif.latex?y%7B_%7Bi%7D%7D%5Cin%20R%5Ek)是![U](algorithm.assets/gif-1598506969806.gif)的第![i](algorithm.assets/gif-1598506970723.gif)行的向量，其中![i=1,2,...,n](algorithm.assets/gif-1598506970459.gif)；

（7）使用k-means算法将新样本点![Y=\left \{ y{_{1}},y{_{2}},...,y{_{n}} \right \}](https://private.codecogs.com/gif.latex?Y%3D%5Cleft%20%5C%7B%20y%7B_%7B1%7D%7D%2Cy%7B_%7B2%7D%7D%2C...%2Cy%7B_%7Bn%7D%7D%20%5Cright%20%5C%7D)聚类成簇![C{_{1}},C{_{2}},...,C{_{k}}](algorithm.assets/gif-1598506970425.gif)；

（8）输出簇![A{_{1}},A{_{2}},...,A{_{k}}](https://private.codecogs.com/gif.latex?A%7B_%7B1%7D%7D%2CA%7B_%7B2%7D%7D%2C...%2CA%7B_%7Bk%7D%7D)，其中，![A{_{i}}=\left \{ j|y{_{j}} \in C{_{i}}\right \}](https://private.codecogs.com/gif.latex?A%7B_%7Bi%7D%7D%3D%5Cleft%20%5C%7B%20j%7Cy%7B_%7Bj%7D%7D%20%5Cin%20C%7B_%7Bi%7D%7D%5Cright%20%5C%7D).

上面就是未标准化的谱聚类算法的描述。也就是先根据样本点计算相似度矩阵，然后计算度矩阵和拉普拉斯矩阵，接着计算拉普拉斯矩阵前k个特征值对应的特征向量，最后将这k个特征值对应的特征向量组成![n*k](algorithm.assets/gif-1598506970587.gif)的矩阵U，U的每一行成为一个新生成的样本点，对这些新生成的样本点进行k-means聚类，聚成k类，最后输出聚类的结果。这就是谱聚类算法的基本思想。相比较PCA降维中取前k大的特征值对应的特征向量，这里取得是前k小的特征值对应的特征向量。但是上述的谱聚类算法并不是最优的，接下来我们一步一步的分解上面的步骤，总结一下在此基础上进行优化的谱聚类的版本。

#### 1.3 谱聚类算法中的重要属性

##### 1.3.1 相似度矩阵介绍

相似度矩阵就是样本点中的任意两个点之间的距离度量，在聚类算法中可以表示为距离近的点它们之间的相似度比较高，而距离较远的点它们的相似度比较低，甚至可以忽略。这里用三种方式表示相似度矩阵：一是![\epsilon](algorithm.assets/gif-1598506970667.gif)-近邻法（![\epsilon](algorithm.assets/gif-1598506970667.gif)-neighborhood graph），二是k近邻法（k-nearest nerghbor graph），三是全连接法（fully connected graph）。下面我们来介绍这三种方法。

**（1）![\epsilon](https://private.codecogs.com/gif.latex?%5Cepsilon)-neighborhood graph：**

​                                   ![s{_{ij}}=||x{_{i}}-x{_{j}}||^2](algorithm.assets/gif-1598506970724.gif)，表示样本点中任意两点之间的欧式距离

用此方法构造的相似度矩阵表示如下：

​                                  ![W{_{ij}}=\begin{cases} 0& \text{ if } s{_{ij}}>\epsilon \\ \epsilon & \text{ if } s{_{ij}}\leq \epsilon \end{cases}](https://private.codecogs.com/gif.latex?W%7B_%7Bij%7D%7D%3D%5Cbegin%7Bcases%7D%200%26%20%5Ctext%7B%20if%20%7D%20s%7B_%7Bij%7D%7D%3E%5Cepsilon%20%5C%5C%20%5Cepsilon%20%26%20%5Ctext%7B%20if%20%7D%20s%7B_%7Bij%7D%7D%5Cleq%20%5Cepsilon%20%5Cend%7Bcases%7D)

该相似度矩阵由于距离近的点的距离表示为![\epsilon](https://private.codecogs.com/gif.latex?%5Cepsilon)，距离远的点距离表示为0，矩阵种没有携带关于数据集的太多的信息，所以该方法一般很少使用，在sklearn中也没有使用该方法。

**（2）k-nearest nerghbor graph：**

由于每个样本点的k个近邻可能不是完全相同的，所以用此方法构造的相似度矩阵并不是对称的。因此，这里使用两种方式表示对称的knn相似度矩阵，第一种方式是如果![v{_{i}}](algorithm.assets/gif-1598506971551.gif)在![v{_{j}}](algorithm.assets/gif-1598506970840.gif)的k个领域中或者![v{_{j}}](algorithm.assets/gif-1598506970840.gif)在![v{_{i}}](algorithm.assets/gif-1598506971551.gif)的k个领域中，则![w{_{ij}}=w{_{ji}}](algorithm.assets/gif-1598506970978.gif)为![v{_{i}}](algorithm.assets/gif-1598506971551.gif)与![v{_{j}}](algorithm.assets/gif-1598506970840.gif)之间的距离，否则为![w{_{ij}}=w{_{ji}}=0](algorithm.assets/gif-1598506970998.gif)；第二种方式是如果![v{_{i}}](algorithm.assets/gif-1598506971551.gif)在![v{_{j}}](algorithm.assets/gif-1598506970840.gif)的k个领域中并且![v{_{j}}](algorithm.assets/gif-1598506970840.gif)在![v{_{i}}](algorithm.assets/gif-1598506971551.gif)的k个领域中，则![w{_{ij}}=w{_{ji}}](algorithm.assets/gif-1598506970978.gif)为![v{_{i}}](algorithm.assets/gif-1598506971551.gif)与![v{_{j}}](algorithm.assets/gif-1598506970840.gif)之间的距离，否则为![w{_{ij}}=w{_{ji}}=0](algorithm.assets/gif-1598506970998.gif)。很显然第二种方式比第一种方式生成的相似度矩阵要稀疏。这两种方式用公式表达如下：

第一种方式：

​                       ![W{_{ij}}=W{_{ji}}=\begin{cases} 0 & \text{ if } x{_{i}} \notin KNN(x{_{j}})\&x{_{j}} \in KNN(x{_{i}}) \\ exp(-\frac{||x{_{i}}-x{_{j}}||^2}{2\sigma ^2}) & \text{ if } x{_{i}} \in KNN(x{_{j}}) |x{_{j}} \in KNN(x{_{i}}) \\ \end{cases}](https://private.codecogs.com/gif.latex?W%7B_%7Bij%7D%7D%3DW%7B_%7Bji%7D%7D%3D%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7B%20if%20%7D%20x%7B_%7Bi%7D%7D%20%5Cnotin%20KNN%28x%7B_%7Bj%7D%7D%29%5C%26x%7B_%7Bj%7D%7D%20%5Cin%20KNN%28x%7B_%7Bi%7D%7D%29%20%5C%5C%20exp%28-%5Cfrac%7B%7C%7Cx%7B_%7Bi%7D%7D-x%7B_%7Bj%7D%7D%7C%7C%5E2%7D%7B2%5Csigma%20%5E2%7D%29%20%26%20%5Ctext%7B%20if%20%7D%20x%7B_%7Bi%7D%7D%20%5Cin%20KNN%28x%7B_%7Bj%7D%7D%29%20%7Cx%7B_%7Bj%7D%7D%20%5Cin%20KNN%28x%7B_%7Bi%7D%7D%29%20%5C%5C%20%5Cend%7Bcases%7D)

第二种方式：

​                      ![W{_{ij}}=W{_{ji}}=\begin{cases} 0 & \text{ if } x{_{i}} \notin KNN(x{_{j}})|x{_{j}} \notin KNN(x{_{i}}) \\ exp(-\frac{||x{_{i}}-x{_{j}}||^2}{2\sigma ^2}) & \text{ if } x{_{i}} \in KNN(x{_{j}}) \& x{_{j}} \in KNN(x{_{i}}) \\ \end{cases}](https://private.codecogs.com/gif.latex?W%7B_%7Bij%7D%7D%3DW%7B_%7Bji%7D%7D%3D%5Cbegin%7Bcases%7D%200%20%26%20%5Ctext%7B%20if%20%7D%20x%7B_%7Bi%7D%7D%20%5Cnotin%20KNN%28x%7B_%7Bj%7D%7D%29%7Cx%7B_%7Bj%7D%7D%20%5Cnotin%20KNN%28x%7B_%7Bi%7D%7D%29%20%5C%5C%20exp%28-%5Cfrac%7B%7C%7Cx%7B_%7Bi%7D%7D-x%7B_%7Bj%7D%7D%7C%7C%5E2%7D%7B2%5Csigma%20%5E2%7D%29%20%26%20%5Ctext%7B%20if%20%7D%20x%7B_%7Bi%7D%7D%20%5Cin%20KNN%28x%7B_%7Bj%7D%7D%29%20%5C%26%20x%7B_%7Bj%7D%7D%20%5Cin%20KNN%28x%7B_%7Bi%7D%7D%29%20%5C%5C%20%5Cend%7Bcases%7D)

**（3）fully connected graph:**

该方法就是在算法描述中的高斯相似度方法，公式如下：

​                     ![W{_{ij}}=W{_{ji}}=\sum_{i=1,j=1}^{n}exp\frac{-||x{_{i}}-x{_{j}}||^2}{2\sigma ^2}](https://private.codecogs.com/gif.latex?W%7B_%7Bij%7D%7D%3DW%7B_%7Bji%7D%7D%3D%5Csum_%7Bi%3D1%2Cj%3D1%7D%5E%7Bn%7Dexp%5Cfrac%7B-%7C%7Cx%7B_%7Bi%7D%7D-x%7B_%7Bj%7D%7D%7C%7C%5E2%7D%7B2%5Csigma%20%5E2%7D)

该方法也是最常用的方法，在sklearn中默认的也是该方法，表示任意两个样本点都有相似度，但是距离较远的样本点之间相似度较低，甚至可以忽略。这里面的参数![\sigma](algorithm.assets/gif-1598506971284.gif)控制着样本点的邻域宽度，即![\sigma](algorithm.assets/gif-1598506971284.gif)越大表示样本点与距离较远的样本点的相似度越大，反之亦然。

##### 1.3.2 拉普拉斯矩阵介绍

对于谱聚类来说最重要的工具就是拉普拉斯矩阵了，下面我们来介绍拉普拉斯矩阵的三种表示方法。

**（1）未标准化的拉普拉斯矩阵：**

未标准化的拉普拉斯矩阵定义如下：

​                   ![L=D-W](https://private.codecogs.com/gif.latex?L%3DD-W)

其中W是上节所说的相似度矩阵，D是度矩阵，在算法描述中有介绍。很显然，W与D都是对称矩阵。

未标准化的拉普拉斯矩阵L满足下面几个性质：

**（a）**对任意一个向量![f(f \in R^n)](https://private.codecogs.com/gif.latex?f%28f%20%5Cin%20R%5En%29)都有：

​                  ![f^TLf=\frac{1}{2} \sum_{i,j=1}^{n}w{_{ij}}(f{_{i}}-f{_{j}})^2](https://private.codecogs.com/gif.latex?f%5ETLf%3D%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bi%2Cj%3D1%7D%5E%7Bn%7Dw%7B_%7Bij%7D%7D%28f%7B_%7Bi%7D%7D-f%7B_%7Bj%7D%7D%29%5E2)

证明如下：

​               ![f^TLf=f^TDf-f^TWf=\sum_{i=1}^{n}d{_{i}}f{_{i}}^2-\sum_{i,j=1}^{n}f{_{i}}f{_{j}}w{_{ij}}](algorithm.assets/gif-1598506971360.gif)

​                          ![=\frac{1}{2}(algorithm.assets/gif-1598506971614.gif)=\frac{1}{2}\sum_{i,j=1}^{n}w{_{ij}}(f{_{i}}-f{_{j}})^2](https://private.codecogs.com/gif.latex?%3D%5Cfrac%7B1%7D%7B2%7D%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dd%7B_%7Bi%7D%7Df%7B_%7Bi%7D%7D%5E2-2%5Csum_%7Bi%2Cj%3D1%7D%5E%7Bn%7Df%7B_%7Bi%7D%7Df%7B_%7Bj%7D%7Dw%7B_%7Bij%7D%7D&plus;%5Csum_%7Bj%3D1%7D%5E%7Bn%7Dd%7B_%7Bj%7D%7Df%7B_%7Bj%7D%7D%5E2%29%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%2Cj%3D1%7D%5E%7Bn%7Dw%7B_%7Bij%7D%7D%28f%7B_%7Bi%7D%7D-f%7B_%7Bj%7D%7D%29%5E2)

**（b）**L是对称的和半正定的，证明如下：

因为![w{_{ij}}\geq 0](https://private.codecogs.com/gif.latex?w%7B_%7Bij%7D%7D%5Cgeq%200)，所以![f^TLf\geq 0](https://private.codecogs.com/gif.latex?f%5ETLf%5Cgeq%200)，所以为半正定矩阵。由于W和D都是对称矩阵，所以L为对称矩阵。

**（c）**L最小的特征值为0，且特征值0所对应的特征向量为全1向量，证明如下：

令![\bar{1}](algorithm.assets/gif-1598506971912.gif)表示![n*1](algorithm.assets/gif-1598506971805.gif)的全1向量，则

​               ![L\cdot \bar{1}=(D-W)\cdot \bar{1}=D\cdot \bar{1}-W\cdot \bar{1}=0\cdot \bar{1}](https://private.codecogs.com/gif.latex?L%5Ccdot%20%5Cbar%7B1%7D%3D%28D-W%29%5Ccdot%20%5Cbar%7B1%7D%3DD%5Ccdot%20%5Cbar%7B1%7D-W%5Ccdot%20%5Cbar%7B1%7D%3D0%5Ccdot%20%5Cbar%7B1%7D)

由D和W的定义可以得出上式。

**（d）**L有n个非负的实数特征值：![0=\lambda{_{1}}\leq \lambda{_{2}}\leq ...\leq \lambda{_{n}}](https://private.codecogs.com/gif.latex?0%3D%5Clambda%7B_%7B1%7D%7D%5Cleq%20%5Clambda%7B_%7B2%7D%7D%5Cleq%20...%5Cleq%20%5Clambda%7B_%7Bn%7D%7D)

**（2）标准化拉普拉斯矩阵**

标准化拉普拉斯矩阵有两种表示方法，一是基于随机游走（Random Walk）的标准化拉普拉斯矩阵![L{_{rw}}](algorithm.assets/gif-1598506972123.gif)和对称标准化拉普拉斯矩阵![L{_{sym}}](algorithm.assets/gif-1598506972214.gif)，定义如下：

​              ![L{_{rw}}=D^{-1}L=I-D^{-1}W](algorithm.assets/gif-1598506972279.gif)

​              ![L{_{sym}}=D^{-1/2}LD^{-1/2}=I-D^{-1/2}WD^{-1/2}](algorithm.assets/gif-1598506972330.gif)

标准化的拉普拉斯矩阵满足如下性质：

**（a）**对任意一个向量![f(f \in R^n)](https://private.codecogs.com/gif.latex?f%28f%20%5Cin%20R%5En%29)都有：

​             ![f^TL{_{rw}}f=f^TL{_{sym}}f=\frac{1}{2}\sum_{i,j=1}^{n}w{_{ij}}(algorithm.assets/gif-1598506973485.gif)^2](https://private.codecogs.com/gif.latex?f%5ETL%7B_%7Brw%7D%7Df%3Df%5ETL%7B_%7Bsym%7D%7Df%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%2Cj%3D1%7D%5E%7Bn%7Dw%7B_%7Bij%7D%7D%28%5Cfrac%7Bf%7B_%7Bi%7D%7D%7D%7B%5Csqrt%7Bd%7B_%7Bi%7D%7D%7D%7D-%5Cfrac%7Bf%7B_%7Bj%7D%7D%7D%7B%5Csqrt%7Bd%7B_%7Bj%7D%7D%7D%7D%29%5E2)

**（b）**当且仅当![\lambda](algorithm.assets/gif-1598506972430.gif)是![L{_{sym}}](algorithm.assets/gif-1598506972214.gif)的特征值，对应的特征向量为![w=D^{1/2}u](algorithm.assets/gif-1598506972508.gif)时，则![\lambda](algorithm.assets/gif-1598506972430.gif)是![L{_{rw}}](algorithm.assets/gif-1598506972123.gif)特征值，对应的特征向量为u；

**（c）**当且仅当![Lu=\lambda Du](https://private.codecogs.com/gif.latex?Lu%3D%5Clambda%20Du)时，![\lambda](https://private.codecogs.com/gif.latex?%5Clambda)是![L{_{rw}}](https://private.codecogs.com/gif.latex?L%7B_%7Brw%7D%7D)的特征值，对应的特征向量为u；

**（d）**0是![L{_{rw}}](algorithm.assets/gif-1598506972123.gif)的特征值，对应的特征向量为![\bar{1}](algorithm.assets/gif-1598506971912.gif)，![\bar{1}](algorithm.assets/gif-1598506971912.gif)为![n*1](algorithm.assets/gif-1598506971805.gif)的全1向量；0也是![L{_{sym}}](algorithm.assets/gif-1598506972214.gif)的特征值，对应的特征向量为![D^{1/2}\bar{1}](algorithm.assets/gif-1598506972818.gif)；

**（e）**![L{_{sym}}](https://private.codecogs.com/gif.latex?L%7B_%7Bsym%7D%7D)和![L{_{rw}}](https://private.codecogs.com/gif.latex?L%7B_%7Brw%7D%7D)是半正定矩阵并且有非负实数特征值：![0=\lambda{_{1}}\leq \lambda{_{2}} \leq ...\leq \lambda{_{n}}](https://private.codecogs.com/gif.latex?0%3D%5Clambda%7B_%7B1%7D%7D%5Cleq%20%5Clambda%7B_%7B2%7D%7D%20%5Cleq%20...%5Cleq%20%5Clambda%7B_%7Bn%7D%7D).

关于各个版本的谱聚类算法的不同之处，就是在于相似度矩阵的计算方式不同和拉普拉斯矩阵的表示方法不同，其它步骤基本相同。下面就来介绍关于谱聚类的两个比较流行的标准化算法。

#### 1.4 标准化谱聚类算法介绍

##### 1.4.1 随机游走拉普拉斯矩阵的谱聚类算法描述

输入：n个样本点![X=\left \{ x{_{1}},x{_{2}},...,x{_{n}} \right \}](https://private.codecogs.com/gif.latex?X%3D%5Cleft%20%5C%7B%20x%7B_%7B1%7D%7D%2Cx%7B_%7B2%7D%7D%2C...%2Cx%7B_%7Bn%7D%7D%20%5Cright%20%5C%7D)和聚类簇的数目k；

输出：聚类簇![A{_{1}},A{_{2}},...,A{_{k}}](https://private.codecogs.com/gif.latex?A%7B_%7B1%7D%7D%2CA%7B_%7B2%7D%7D%2C...%2CA%7B_%7Bk%7D%7D)

（1）计算![n*n](https://private.codecogs.com/gif.latex?n*n)的相似度矩阵W；

（2）计算度矩阵D；

（3）计算拉普拉斯矩阵![{\color{Red} L{_{rw}}=D^{-1}L=D^{-1}(D-W)}](https://private.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20L%7B_%7Brw%7D%7D%3DD%5E%7B-1%7DL%3DD%5E%7B-1%7D%28D-W%29%7D)；

（4）计算![L{_{rw}}](https://private.codecogs.com/gif.latex?L%7B_%7Brw%7D%7D)的特征值，将特征值从小到大排序，取前k个特征值，并计算前k个特征值的特征向量![u{_{1}},u{_{2}},...,u{_{k}}](https://private.codecogs.com/gif.latex?u%7B_%7B1%7D%7D%2Cu%7B_%7B2%7D%7D%2C...%2Cu%7B_%7Bk%7D%7D)；

（5）将上面的k个列向量组成矩阵![U=\left \{ u{_{1}},u{_{2}},...,u{_{k}} \right \}](https://private.codecogs.com/gif.latex?U%3D%5Cleft%20%5C%7B%20u%7B_%7B1%7D%7D%2Cu%7B_%7B2%7D%7D%2C...%2Cu%7B_%7Bk%7D%7D%20%5Cright%20%5C%7D)，![U\in R^{n*k}](https://private.codecogs.com/gif.latex?U%5Cin%20R%5E%7Bn*k%7D)；

（6）令![y{_{i}}\in R^k](https://private.codecogs.com/gif.latex?y%7B_%7Bi%7D%7D%5Cin%20R%5Ek)是![U](https://private.codecogs.com/gif.latex?U)的第![i](https://private.codecogs.com/gif.latex?i)行的向量，其中![i=1,2,...,n](https://private.codecogs.com/gif.latex?i%3D1%2C2%2C...%2Cn)；

（7）使用k-means算法将新样本点![Y=\left \{ y{_{1}},y{_{2}},...,y{_{n}} \right \}](https://private.codecogs.com/gif.latex?Y%3D%5Cleft%20%5C%7B%20y%7B_%7B1%7D%7D%2Cy%7B_%7B2%7D%7D%2C...%2Cy%7B_%7Bn%7D%7D%20%5Cright%20%5C%7D)聚类成簇![C{_{1}},C{_{2}},...,C{_{k}}](https://private.codecogs.com/gif.latex?C%7B_%7B1%7D%7D%2CC%7B_%7B2%7D%7D%2C...%2CC%7B_%7Bk%7D%7D)；

（8）输出簇![A{_{1}},A{_{2}},...,A{_{k}}](https://private.codecogs.com/gif.latex?A%7B_%7B1%7D%7D%2CA%7B_%7B2%7D%7D%2C...%2CA%7B_%7Bk%7D%7D)，其中，![A{_{i}}=\left \{ j|y{_{j}} \in C{_{i}}\right \}](https://private.codecogs.com/gif.latex?A%7B_%7Bi%7D%7D%3D%5Cleft%20%5C%7B%20j%7Cy%7B_%7Bj%7D%7D%20%5Cin%20C%7B_%7Bi%7D%7D%5Cright%20%5C%7D).

##### 2.4.2 对称拉普拉斯矩阵的谱聚类算法描述

输入：n个样本点![X=\left \{ x{_{1}},x{_{2}},...,x{_{n}} \right \}](https://private.codecogs.com/gif.latex?X%3D%5Cleft%20%5C%7B%20x%7B_%7B1%7D%7D%2Cx%7B_%7B2%7D%7D%2C...%2Cx%7B_%7Bn%7D%7D%20%5Cright%20%5C%7D)和聚类簇的数目k；

输出：聚类簇![A{_{1}},A{_{2}},...,A{_{k}}](https://private.codecogs.com/gif.latex?A%7B_%7B1%7D%7D%2CA%7B_%7B2%7D%7D%2C...%2CA%7B_%7Bk%7D%7D)

（1）计算![n*n](https://private.codecogs.com/gif.latex?n*n)的相似度矩阵W；

（2）计算度矩阵D；

（3）计算拉普拉斯矩阵![{\color{Red} L{_{rsym}}=D^{-1/2}LD^{-1/2}=D^{-1/2}(D-W)D^{-1/2}}](https://private.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20L%7B_%7Brsym%7D%7D%3DD%5E%7B-1/2%7DLD%5E%7B-1/2%7D%3DD%5E%7B-1/2%7D%28D-W%29D%5E%7B-1/2%7D%7D)；

（4）计算![L{_{rw}}](https://private.codecogs.com/gif.latex?L%7B_%7Brw%7D%7D)的特征值，将特征值从小到大排序，取前k个特征值，并计算前k个特征值的特征向量![u{_{1}},u{_{2}},...,u{_{k}}](https://private.codecogs.com/gif.latex?u%7B_%7B1%7D%7D%2Cu%7B_%7B2%7D%7D%2C...%2Cu%7B_%7Bk%7D%7D)；

（5）将上面的k个列向量组成矩阵![U=\left \{ u{_{1}},u{_{2}},...,u{_{k}} \right \}](https://private.codecogs.com/gif.latex?U%3D%5Cleft%20%5C%7B%20u%7B_%7B1%7D%7D%2Cu%7B_%7B2%7D%7D%2C...%2Cu%7B_%7Bk%7D%7D%20%5Cright%20%5C%7D)，![U\in R^{n*k}](https://private.codecogs.com/gif.latex?U%5Cin%20R%5E%7Bn*k%7D)；

（6）令![y{_{i}}\in R^k](https://private.codecogs.com/gif.latex?y%7B_%7Bi%7D%7D%5Cin%20R%5Ek)是![U](https://private.codecogs.com/gif.latex?U)的第![i](https://private.codecogs.com/gif.latex?i)行的向量，其中![i=1,2,...,n](https://private.codecogs.com/gif.latex?i%3D1%2C2%2C...%2Cn)；

（7）对于![{\color{Red} i=1,2,...,n}](https://private.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20i%3D1%2C2%2C...%2Cn%7D)，将![{\color{Red} y{_{i}}\in R^k}](https://private.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20y%7B_%7Bi%7D%7D%5Cin%20R%5Ek%7D)依次单位化，使得![{\color{Red} |y{_{i}}|=1}](https://private.codecogs.com/gif.latex?%7B%5Ccolor%7BRed%7D%20%7Cy%7B_%7Bi%7D%7D%7C%3D1%7D)；

（8）使用k-means算法将新样本点![Y=\left \{ y{_{1}},y{_{2}},...,y{_{n}} \right \}](https://private.codecogs.com/gif.latex?Y%3D%5Cleft%20%5C%7B%20y%7B_%7B1%7D%7D%2Cy%7B_%7B2%7D%7D%2C...%2Cy%7B_%7Bn%7D%7D%20%5Cright%20%5C%7D)聚类成簇![C{_{1}},C{_{2}},...,C{_{k}}](https://private.codecogs.com/gif.latex?C%7B_%7B1%7D%7D%2CC%7B_%7B2%7D%7D%2C...%2CC%7B_%7Bk%7D%7D)；

（9）输出簇![A{_{1}},A{_{2}},...,A{_{k}}](https://private.codecogs.com/gif.latex?A%7B_%7B1%7D%7D%2CA%7B_%7B2%7D%7D%2C...%2CA%7B_%7Bk%7D%7D)，其中，![A{_{i}}=\left \{ j|y{_{j}} \in C{_{i}}\right \}](https://private.codecogs.com/gif.latex?A%7B_%7Bi%7D%7D%3D%5Cleft%20%5C%7B%20j%7Cy%7B_%7Bj%7D%7D%20%5Cin%20C%7B_%7Bi%7D%7D%5Cright%20%5C%7D).

上面两个标准化拉普拉斯算法加上未标准化拉普拉斯算法这三个算法中，主要用到的技巧是将原始样本点![x{_{i}}](algorithm.assets/gif-1598506973661.gif)转化为新的样本点![y{_{i}}](algorithm.assets/gif-1598506973736.gif)，然后再对新样本点使用其它的聚类算法进行聚类，在这里最后一步用到的聚类算法不一定非要是KMeans算法，也可以是其它的聚类算法，具体根据实际情况而定。在sklearn中默认是使用KMeans算法，但是由于KMeans聚类对初始聚类中心的选择比较敏感，从而导致KMeans算法不稳定，进而导致谱聚类算法不稳定，所以在sklearn中有另外一个可选项是'discretize'，该算法对初始聚类中心的选择不敏感。

### 2. 谱聚类算法的优缺点

#### 2.1 优点

（1）当聚类的类别个数较小的时候，谱聚类的效果会很好，但是当聚类的类别个数较大的时候，则不建议使用谱聚类；

（2）谱聚类算法使用了降维的技术，所以更加适用于高维数据的聚类；

（3）谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法（比如K-Means）很难做到

（4）谱聚类算法建立在谱图理论基础上，与传统的聚类算法相比，它具有能在任意形状的样本空间上聚类且收敛于全局最优解

#### 2.2 缺点

（1）谱聚类对相似度图的改变和聚类参数的选择非常的敏感；

（2）谱聚类适用于均衡分类问题，即各簇之间点的个数相差不大，对于簇之间点个数相差悬殊的聚类问题，谱聚类则不适用；

## 插值

### 1、最近邻插值算法（零阶插值算法）

目标图像B（X,Y）通过同时求得源图像A（x+u,y+v）（u，v是<=1的小数），则对应在源图像上的坐标为A（x,y）=A（i,j）,所以要找邻近的4个像素点：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812180618696-198601981.png)

> 如果 i+u, j+v(i落在 A区，即 u<0.5,v<0.5，则将左上角象素的灰度值赋给待求象素，同理落在B区则赋予右上角的象素灰度值，落在C区则赋予左下角象素的灰度值，落在D区则赋予右下角象素的灰度值。
> 最近邻插值法计算量较小，但可能会造成生的图像灰度上的不连续，在变化地方可能出现明显锯齿状。

近邻取样插值缩放简单、速度快，但是缩放出的图片质量比较差，当图片放大时，缺少的像素通过直接使用与之最近原有颜色生成，也就是说照搬旁边的像素这样做结果产生了明显可见的锯齿。效果不好的根源就是其简单的最临近插值方法引入了严重的图像失真，比如，当由目标图的坐标反推得到的源图的的坐标是一个浮点数的时候，采用了四舍五入的方法，直接采用了和这个浮点数最接近的象素的值，这种方法是很不科学的。

### 2、双线性（一阶插值法）

经过三次插值才能得到最终结果，是对最近邻的改进。先对两水平方向进行一阶线性插值，然后再在垂直方向上进行一阶线性插值。能创造出比双线性插值更平滑的图像边缘。

#### 单线性插值

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812170655849-206079103.png)

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812170723190-1005441684.png)

相当于在y=y0和y=y1这两个值上做了线性的插值。

#### 双线性插值

双线性插值是**有两个变量的插值函数的线性插值扩展**，在两个方向分别进行一次线性插值。

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812171004306-1999143646.png)

假如想得到未知函数f在p点的值，已经知道了f在Q11 = (x1, y1)、Q12 = (x1, y2), Q21(x2, y1) 以及 Q22(x2, y2) 四个点的值。

首先在x方向进行两次线性插值：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812171453555-431602158.png)

然后在 y 方向进行一次线性插值：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812171520131-1530335642.png)

图像双线性插值只会用相邻的4个点，opencv中用了一些优化手段，比如用整数计算代替float，源图像和目标图像几何中心的对齐。

假设源图像A大小为m*n，像素坐标为（x，y），缩放后的目标图像大小是M*N，依次求B（X，Y）每一个像素点的值，先找到B（X,Y）对应在A（x，y）的坐标：

x = X * （m/M）

y = Y * （n/N）

中心对齐(OpenCV也是如此)： 

```
SrcX=(dstX+0.5) * (srcWidth/dstWidth) -0.5` 
`SrcY=(dstY+0.5) * (srcHeight/dstHeight)-0.5
```

源图像和目标图像的原点（0，0）均选择左上角，假设你需要将一幅5x5的图像缩小成3x3，那么源图像和目标图像各个像素之间的对应关系如下。如果没有这个两个图像的几何中心对齐，根据基本公式去算，就会发现源图像中最右边和最下边上的像素没有参与运算，输出图像的像素点的灰度值相对于源图像偏，就会得到左边这样的结果；而用了对齐，就会得到右边的结果： 

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812172309233-1070634153.png)

#### 效果分析

效果好于最近邻插值，计算量稍大，放缩后图像质量提高，基本克服了最近邻插值灰度值不连续的缺点，但是由于只考虑了相邻的4个点，没有考虑各个点之间灰度值的变化率的影响，因此具有低通滤波的作用，知道图像的高频分量受到影响，图像边缘在一定程度上变得模糊。

### 3、立方卷积插值算法（双三次、双立方）

是对双线性插值的改进，是一种较为复杂的插值方式，它不仅考虑到相邻的4*4像素点灰度值的影响，还考虑到它们灰度值变化率的影响。

#### 卷积插值公式

假设P（x+u，y+v）点就是（x，y）对应在目标图像的位置，双立方差值就是通过bicubic基函数得到目标像素点周围的16个相邻像素目标像素点P的影响因子，该**基函数（卷积插值公式）**是：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812202457104-1322614670.png)

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812202548640-1688214923.png)

> a=-0.5时比较合适
>
> x位在目标图像中相邻的16个像素到P的距离。

##### a=-1时

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812214528352-340972767.png)

此时逼近的函数是y = sin(x*π)/(x*π)：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812214622248-609031176.png)

##### 当a=-0.5[ ](https://en.wikipedia.org/wiki/Cubic_Hermite_spline)

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812214652390-1360687483.png)

此时逼近[三次Hermite样条](https://en.wikipedia.org/wiki/Cubic_Hermite_spline)：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812214739510-1666317492.png)

##### 对比图

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812214834599-1752280695.png)

##### 过程

假设源图像A大小为m*n，像素坐标为（x，y），缩放后的目标图像大小是M*N，依次求B（X，Y）每一个像素点的值，先找到B（X,Y）对应在A的坐标（x+u,y+v），因为计算出来的不可能肯定是整数，所以这样表示，然后找找到最接近的点就是A(x,y)，最终B(X,Y)的像素值就是由A(x,y)附近的16个像素点来决定，这十六个像素点的范围是：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812213831282-1559227751.png)

 

设像素点的像素值的函数是f(x,y)，那么目标图像B中对应的像素点（X,Y）的像素值为F(x+u,y+v)：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812213928718-41389261.png)

> S（x）就是卷积插值公式W（x）。
>
> a取-0.5

矩阵形式：

![img](https://img2018.cnblogs.com/blog/1476416/201908/1476416-20190812214030591-721943071.png)

#### 效果分析

立方卷积插值不仅考虑到周围四个直接相邻像素点灰度值的影响，还考虑到它们灰度值变化率的影响。因此克服了前两种方法的不足之处，能够产生比双线性插值更为平滑的边缘，计算精度很高，处理后的图像像质损失最少，效果是最佳的。

### 4.多项式插值（得到的是经过所有点的一个插值函数）**

**①一般多项式插值**

**![img](http://img.blog.itpub.net/blog/2019/09/13/75ed546ff6cc9e5d.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/2d8c0a5fa8cc0647.png?x-oss-process=style/bb)**

**②.拉格朗日插值法**

**![img](http://img.blog.itpub.net/blog/2019/09/13/cb7bd49e6aeea139.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/7f116d5d6b3d9eeb.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/3fb5ead4a69f1aef.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/832aa8a555ceb6a6.png?x-oss-process=style/bb)**

**③.牛顿插值法**

**![img](http://img.blog.itpub.net/blog/2019/09/13/3dab39aba3c4b997.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/d5a7631a1ca9e943.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/13d37a59bd0ad240.png?x-oss-process=style/bb)**

多项式插值存在的问题：

龙格现象：当函数的次数过高时，xi加大一点对函数值的影响就会很大。

**![img](http://img.blog.itpub.net/blog/2019/09/13/f0212a776f0b293e.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/492f94d6e338a615.png?x-oss-process=style/bb)**

为了解决龙格现象，引入了分段插值。

**2.分段插值（非得到一个插值函数，而是用很多分段插值函数求每个分段上的xi的函数值）**

**![img](http://img.blog.itpub.net/blog/2019/09/13/c6d0a0c559a19b6c.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/02e5eb90b46517a4.png?x-oss-process=style/bb)**

**①埃尔米特插值：不仅函数值相等，而且一阶导数相等**

**![img](http://img.blog.itpub.net/blog/2019/09/13/3f74eb7102d11f7b.png?x-oss-process=style/bb)**

**![img](http://img.blog.itpub.net/blog/2019/09/13/6e92db6087f4fb78.png?x-oss-process=style/bb)**

 

**②分段三次埃尔米特插值**

**![img](http://img.blog.itpub.net/blog/2019/09/13/fe62f824ae311160.png?x-oss-process=style/bb)**

**③三次样条插值：二阶导数连续可微**

**![img](http://img.blog.itpub.net/blog/2019/09/13/888ebb9ecc6d0fa7.png?x-oss-process=style/bb)**

对比：

**![img](http://img.blog.itpub.net/blog/2019/09/13/97dedc91a3946cb2.png?x-oss-process=style/bb)**

 

**n维插值问题：（同一维插值）**

**![img](http://img.blog.itpub.net/blog/2019/09/13/48ab95be8eea7612.png?x-oss-process=style/bb)**

 

## 拟合

## 降维

### PCA

![【机器学习】降维——PCA（非常详细）](algorithm.assets/v2-e47296e78fff3d97eea11d0657ddcb81_1440w.jpg)

#### 算法基础

#####  协方差和散度矩阵

样本均值：

![img](algorithm.assets/20180609143801402.png)

样本方差：

![img](algorithm.assets/20180609143835695.png)

样本X和样本Y的协方差：

![img](algorithm.assets/20180609143922240.png)

由上面的公式，我们可以得到以下结论：

(1) 方差的计算公式是针对一维特征，即针对同一特征不同样本的取值来进行计算得到；而协方差则必须要求至少满足二维特征；方差是协方差的特殊情况。

(2) 方差和协方差的除数是n-1,这是为了得到方差和协方差的无偏估计。

协方差为正时，说明X和Y是正相关关系；协方差为负时，说明X和Y是负相关关系；协方差为0时，说明X和Y是相互独立。Cov(X,X)就是X的方差。当样本是n维数据时，它们的协方差实际上是协方差矩阵(对称方阵)。例如，对于3维数据(x,y,z)，计算它的协方差就是：

![img](algorithm.assets/20180609143959918.png)

散度矩阵定义为：

![img](algorithm.assets/20180609144105125.png)

对于数据X的散度矩阵为![img](algorithm.assets/20180609144136638.png)。其实协方差矩阵和散度矩阵关系密切，散度矩阵就是协方差矩阵乘以（总数据量-1）。因此它们的特征值和特征向量是一样的。这里值得注意的是，散度矩阵是SVD奇异值分解的一步，因此PCA和SVD是有很大联系。

##### 特征值分解矩阵原理

(1) 特征值与特征向量

如果一个向量v是矩阵A的特征向量，将一定可以表示成下面的形式：

![img](algorithm.assets/20180609144203490.png)

其中，λ是特征向量v对应的特征值，一个矩阵的一组特征向量是一组正交向量。

(2) 特征值分解矩阵

对于矩阵A，有一组特征向量v，将这组向量进行正交化单位化，就能得到一组正交单位向量。特征值分解，就是将矩阵A分解为如下式：

![img](algorithm.assets/20180609144238718.png)

其中，Q是矩阵A的特征向量组成的矩阵，![\Sigma](algorithm.assets/equation.svg)则是一个对角阵，对角线上的元素就是特征值。

##### SVD分解矩阵原理

奇异值分解是一个能适用于任意矩阵的一种分解的方法，对于任意矩阵A总是存在一个奇异值分解：

![img](algorithm.assets/20180609144311809.png)

假设A是一个m*n的矩阵，那么得到的U是一个m*m的方阵，U里面的正交向量被称为左奇异向量。Σ是一个m*n的矩阵，Σ除了对角线其它元素都为0，对角线上的元素称为奇异值。![img](algorithm.assets/20180609144333940.png)是v的转置矩阵，是一个n*n的矩阵，它里面的正交向量被称为右奇异值向量。而且一般来讲，我们会将Σ上的值按从大到小的顺序排列。

SVD分解矩阵A的步骤：

(1) 求![img](algorithm.assets/20180609144409595.png)的特征值和特征向量，用单位化的特征向量构成 U。

(2) 求![img](algorithm.assets/20180609144423413.png)的特征值和特征向量，用单位化的特征向量构成 V。

(3) 将![img](algorithm.assets/20180609144409595.png)或者![img](algorithm.assets/20180609144423413.png)的特征值求平方根，然后构成 Σ。

#### 算法流程

##### (1) 基于特征值分解协方差矩阵实现PCA算法

输入：数据集![img](algorithm.assets/20180609144504394.png)，需要降到k维。

1) 去平均值(即去中心化)，即每一位特征减去各自的平均值。、

2) 计算协方差矩阵![img](algorithm.assets/20180609144531611.png),注：这里除或不除样本数量n或n-1,其实对求出的特征向量没有影响。

3) 用特征值分解方法求协方差矩阵![img](algorithm.assets/20180609144531611.png)的特征值与特征向量。

4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为行向量组成特征向量矩阵P。

5) 将数据转换到k个特征向量构建的新空间中，即Y=PX。

##### (2) 基于SVD分解协方差矩阵实现PCA算法

输入：数据集![img](algorithm.assets/20180609145316667.png)，需要降到k维。

1) 去平均值，即每一位特征减去各自的平均值。

2) 计算协方差矩阵。

3) 通过SVD计算协方差矩阵的特征值与特征向量。

4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵。

5) 将数据转换到k个特征向量构建的新空间中。

#### 算法优缺点

**优点：**

（1）使得数据集更易使用；

（2）降低算法的计算开销；

（3）去除噪声；

（4）使得结果容易理解；

（5）完全无参数限制。

**缺点：**

（1）如果用户对观测对象有一定的先验知识，掌握了数据的一些特征，却无法通过参数化等方法对处理过程进行干预，可能会得不到预期的效果，效率也不高；

（2） 特征值分解有一些局限性，比如变换的矩阵必须是方阵，因此sklearn采用SVD求特征值与特征向量；

（3） 在非高斯分布情况下，PCA方法得出的主元可能并不是最优的。



### KPCA

KPCA是一种非线性主元分析方法，用于降维。主要思想：通过某种事先选择的非线性映射函数Ф将输入矢量X映射到一个高维线性特征空间F之中，然后在空间F中使用PCA方法计算主元成分，核主成分分析最主要是非线性映射函数Ф的选取。

#### 算法步骤：

Step 1. 数据标准化处理。

Step 2. 求核矩阵K，使用核函数来实现将原始数据由数据空间映射到特征空间。采用的核函数为径向基核函数，公式为：![img](algorithm.assets/20150907165603522.png)

Step 3. 中心化核矩阵Kc，用于修正核矩阵。公式为：  ![img](algorithm.assets/20150907165729327.png)

其中， 为N×N的矩阵，每一个元素都为1/N

Step 4. 计算矩阵KC的特征值 ，对应的特征向量为![img](algorithm.assets/20150907165931213.png) 。特征值决定方差的大小![img](algorithm.assets/20150907165956197.png)，也就是说特征值越大所蕴含的有用信息越多，因此按特征值降序排序得 ，特征向量相应调整 。

Step 5. 通过施密特正交化方法，正交化并单位化特征向量，得到![img](algorithm.assets/20150907170104785.png) 。

Step 6. 计算特征值的累计贡献率![img](algorithm.assets/20150907170128882.png) ，根据给定的贡献率要求p，如果rt>p，则选取前t个主分量![img](algorithm.assets/20150907170149617.png)，作为降维后的数据。



### LDA

**LDA原理与流程**

![0?wx_fmt=png](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHAMEdxoVUETWc5TDf4KyBW4U3s6FpZmj85OSbZAWpBXick6zrCyFRLw6vyUlS4HibLdbK7ZiaxTP5ic7Q/0?wx_fmt=png)

 

![0?wx_fmt=png](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHAMEdxoVUETWc5TDf4KyBW4DwUEFNNiaLIfAPg2dcpP9htbicYBDN4iaVn4KFeFQh77VyaySibmYicYJibw/0?wx_fmt=png)

 

![0?wx_fmt=png](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHAMEdxoVUETWc5TDf4KyBW46EEBkPl2NAYD8aDHCREMzJl3U2aZGBN5TK4XVYibu8fD3LbwkicQWoAQ/0?wx_fmt=png)

 

![0?wx_fmt=png](https://ss.csdn.net/p?http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHAMEdxoVUETWc5TDf4KyBW4j1whYKNf2icErpLDOg2iaZkRcxAhS9sXL4hrBxSXJhibCZrLsmZM4RxNg/0?wx_fmt=png)

#### 算法流程

输入：数据集D={(x1,y1),(x2,y2),...,((xm,ym))},其中任意样本xi为n维向量，yi∈{C1,C2,...,Ck}，降维到的维度d。

输出：降维后的样本集$D′$

1.  计算类内散度矩阵Sw
2.  计算类间散度矩阵Sb
3. 计算矩阵$Sw^{−1}*Sb$
4. 计算 $Sw^{−1}*Sb$ 的最大的d个特征值和对应的d个特征向量(w1,w2,...wd),得到投影矩阵W
5. 对样本集中的每一个样本特征xi,转化为新的样本$zi=WT*xi$
6. 得到输出样本集

以上就是使用LDA进行降维的算法流程。实际上LDA除了可以用于降维以外，还可以用于分类。一个常见的LDA分类基本思想是假设各个类别的样本数据符合高斯分布，这样利用LDA进行投影后，可以利用极大似然估计计算各个类别投影数据的均值和方差，进而得到该类别高斯分布的概率密度函数。当一个新的样本到来后，我们可以将它投影，然后将投影后的样本特征分别带入各个类别的高斯分布概率密度函数，计算它属于这个类别的概率，最大的概率对应的类别即为预测类别。

优点:

(1) 计算速度快。

(2) 充分利用了先验知识。

(3) LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。

缺点:

(1) 当数据不是高斯分布时候，效果不好，PCA也是。

(2)LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。当然目前有一些LDA的进化版算法可以绕过这个问题。

(3) LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好.

(4) LDA可能过度拟合数据。

特点:

降维之后的维数最多为类别数-1。所以当数据维度很高，但是类别数少的时候，算法并不适用。



### MDS

![1598600630387](algorithm.assets/1598600630387.png)

### LLE

![1598600704290](algorithm.assets/1598600704290.png)

## 特征提取

## 数据预测

## 分类

## 异常检测

## 跟踪

## 滤波

## 控制

## 关联

## 推荐

## 优化

