
# 使用
OCR服务以web的方式对外提供接口。 推荐使用我们发布的docker镜像，~~点击下载~~

## 在docker中启动服务
**请务必暴露docker容器的`555`端口**

运行 `python /usr/local/src/universal_cnn/app.py`

> 预训练模型已在docker镜像中配置好

## 使用OCR服务
服务接口为标准GET请求：
`http://[host]:555/?path=[your_image_path]`

`your_image_path`是需要做识别的图片路径，**务必确保它已经位于容器中**

> **不包含文件上传的功能**，我们不对“使用何种方式上传文件?”，“文件上传到哪里？”，“识别后是否删除文件？”等相关问题提供统一的解决方案
> 这些问题由使用者来解决

### 示例
使用`wget`调用服务
``
