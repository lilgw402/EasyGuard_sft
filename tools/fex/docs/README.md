# 生成文档

1. 从注释生成文档：
```bash
cd docs
sphinx-apidoc -f -o source/api/ ../fex/ ../fex/*/*_test*
```

2. 编译：
```bash
cd docs
make clean
make html
```

3. 启动服务器：
```bash
cd build/html
python -m http.server
```