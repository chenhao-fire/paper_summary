git 使用方式

1、git branch -m

对已经存在的branch重名；其中：

-m，--move   Move/rename a branch and the corresponding reflog.

-M Move/rename a branch even if the new branch name already exists.

如： git branch -m temp2 hello; 表示将temp2的branch命名为hello

2、git branch -D hello

删除hello这个分支

3、git push -u

将本地分支mybranch1与远程仓库origin里的分支mybranch1建立关联；

如： git push -u origin mybranch1

在远程分支存在mybranch1的情况下等同于git branch --set-upstream-to=origin/mybranch1 mybranch1

两种都能达到目的，但1方法更通用。如果远程库中没有mybranch1分支，则2方法不行。

总结：git push -u origin mybranch1 相当于 git push origin mybranch1 + git branch --set-upsteam-to=origin/mybranch1 mybranch1