
from PIL import Image, ImageDraw, ImageFont

def draw_similarity_results(target:Image.Image, others:list[Image.Image], similarities:list[float]):
    width, height = target.size
    total_width = width * (1 + len(others))
    new_img = Image.new('RGB', (total_width, height))

    new_img.paste(target, (0, 0))

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(new_img)

    # 遍历其他图像并依次粘贴到新图像中，并在上面绘制相似度值
    for i, (other, sim) in enumerate(zip(others, similarities)):
        # 计算当前图像放置位置
        left = width * (i + 1)
        other = other.resize((width, height))  # 确保所有图片大小一致
        new_img.paste(other, (left, 0))

        # 在图像上方绘制相似度值
        text = f"{sim:.2f}"
        # 使用font对象获取文本尺寸
        text_x = left + 5
        text_y = 10  # 距离顶部的距离
        draw.text((text_x, text_y), text, fill="red" if sim < 0.7 else "green", font=font,stroke_fill="white",stroke_width=2)

    return new_img

def merge_images_vertically(image_list:list[Image.Image])->Image.Image:
    """
    将一组PIL图片垂直合并为一张图片。
    
    :param image_list: 包含PIL Image对象的列表，这些图像将被垂直堆叠。
    :return: 返回一个垂直合并后的PIL Image对象。
    """
    # 计算所有图像组合后的宽度（取最大宽度）
    max_width = max([img.width for img in image_list])
    
    # 计算组合后图像的高度（所有图像高度之和）
    total_height = sum([img.height for img in image_list])

    # 创建一个新的空白图像，用于放置合并后的结果
    merged_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0 # 这个变量用来跟踪下一个图像应该放置的位置
    for img in image_list:
        # 粘贴当前图像到合并图像上
        merged_image.paste(img, (0, y_offset))
        # 更新y_offset，为下一个图像留出空间
        y_offset += img.height

    return merged_image