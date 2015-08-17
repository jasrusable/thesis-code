from images import Image, FeaturizedImage
from jason_cvs import JasonCV


object_image = Image(path="../images/model_1/object.png")
scene_image = Image(path="../images/model_1/scene.png")


jason_cv = JasonCV()
sifted_object_image = jason_cv.get_sifted_image(object_image)
sifted_scene_image = jason_cv.get_sifted_image(scene_image)

matches = jason_cv.run_flan(sifted_object_image, sifted_scene_image)
