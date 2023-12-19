# 浙江大学 2023 冬计算机视觉
本仓库为浙江大学 2023 冬《计算机视觉》课程仓库，主要面向[潘纲](https://person.zju.edu.cn/gpan)老师教学班，也欢迎其他同学参考。

如果对本仓库有任何意见欢迎提出 issue，或通过其他任何方式联系我。

建议使用[在线文档](https://zhoutimemachine.github.io/2023_CV/)，或者如果希望本地部署，可以首先安装 mkdocs 支持
```
pip install mkdocs
pip install mkdocs-material
pip install mkdocs-heti-plugin
```

打开实时渲染服务（默认端口 8000）
```
mkdocs serve
```

如果顺利的话，在浏览器中输入 `127.0.0.1:8000` 就可以本地预览了。但是如果 8000 端口被占用，可能需要指定一个新的端口，以 8001 为例：
```
mkdocs serve -a 127.0.0.1:8001
```

此时就需要使用 `127.0.0.1:8001` 进行本地预览了。

本仓库实时更新，因此如果使用本地预览，可以时不时 `git pull` 以获取最新版本。