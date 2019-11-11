
universal_cnn打包exe注意事项

代码主体依旧是universal_cnn，添加windows前端和对应的输入输出后，使用pyinstaller打包，以下主要为pyinstaller的踩坑记录

---

打包过程

1. 使用管理员权限打开命令模式，并切换到universal_cnn所在目录
2. 使用pyinstaller -F path_input.app得到一个spec文件（会报错，但只需要这个spec文件即可）
3. 编辑spec文件解决递归问题后，输入chcp 65001转utf8代码页，执行pyinstaller -F path_input.spec
4. 将项目中的静态文件复制到exe的同目录下（执行失败或者版本问题见下方Trouble Shooting）

依赖

pyinstaller要求所有引入包都必须在main函数所在文件中声明，对应的，如果下下来，有依赖不全的话可以直接查看main函数所在包

TODO

1. 去掉项目中与OCR执行无关的部分，缩小文件大小
2. 优化前端，添加文件显示部分，使得文件增删变得更为简单自由，而不是需要通过查看超长的字符串去查看文件是否遗漏，或者每次都得重选来增删所选文件

Trouble Shooting

1. 递归

在打包的时候出现RecursionError: maximum recursion depth exceeded

1. pyinstaller filename.py
会在报错后产生一个对应的.spec文件
2. 在这个文件的开头加上限制递归次数的代码

    import sys
    sys.setrecursionlimit(5000)

3. 然后运行.spec文件

    pyinstaller filename.spec

by StackOverflow

2. 乱码

在打包的时候会抛出UnicodeDecodeError: 'utf-8' codec can't decode byte * in position:invalid continuation b

输入命令

    chcp 65001

切换到utf8代码页再执行打包即可
by CSDN

3. pyinstaller failed to execute script

法一（推荐）：

    # 使用完下面这条指令之后,打开exe,提示failed to execute script
    pyinstaller -Fw pachonggui.py
    # 然后执行下面这条执行,会在list下生成一个目录,进入该目录,用**命令行**执行该exe,就会看到错误了
    pyinstaller -D pachonggui.py

法二：

命令执行完毕之后 build\pachonggui\warnpachonggui.txt,上面会记载有哪些依赖缺失，但是上面的依赖缺失仅供参考，其实不一定需要，主要解决方式还是法一

4. Cython h5py的版本冲突问题

第三条运行报错可能出现AttributeError: type object 'h5py.h5r.Reference' has no attribute 'reduce_cython'。是python 3.6的版本兼容问题。在网上查了很多个解决版本，最后我成功的版本是h5py==2.8.0, cython==0.27。

此外，还可能出现Warning! HDF5 library version mismatched error，同样是版本问题，继续尝试替换版本来解决吧。

5. 静态文件的路径问题

通常，为了访问的方便和移植性，访问静态文件都是使用相对路径，但是使用pyinstaller打包成exe就会出现相对路径没有办法访问，最后使用的是妥协式的解决办法，把静态文件拿出来和.exe放在同一路径下，然后通过绝对路径访问静态文件。在universal_cnn中主要有三个静态文件文件夹：
all: 存储的应该是训练得到的检查点和相关数据

configs: 存储的是一些路径和训练相关的参数设置

static: 存储图片截取后的部分，供识别，可以通过auxliary参数设置是否产生该参数，通过box选择截取哪部分图片

