import numpy as np
import matplotlib.pyplot as plt
def height(x, y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
# 将原始数据变成网格数据
X,Y = np.meshgrid(x,y)
# 填充颜色
plt.contourf(X,Y,height(X,Y),8,alpha=0.75,cmap=plt.cm.hot)
# add contour lines
C = plt.contour(X,Y,height(X,Y),8,color='black',lw=0.5)
# 显示各等高线的数据标签cmap=plt.cm.hot
plt.clabel(C,inline=True,fontsize=10)

plt.show()