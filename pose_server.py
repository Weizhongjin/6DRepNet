import os
from src.server import BaseHandler, deploy
from src.sixDRepNet_Estimator import PoseEstimator

IMG_SIZE = 12
cur_path = os.path.dirname(__file__)  # current path
__PAGE__ = os.path.join(cur_path, 'index/')  # storage path for demo page

class PoseHandler(BaseHandler):
    def init(self,pose_estimator: PoseEstimator ):
        self.pose_estimator = pose_estimator
    def algorithm(self, req, image):
        image_name = req.get("image_name")
        result = self.pose_estimator.predict(image)
        final_res = {
            "image_name" : image_name,
            "face_name" : len(result),
            "face_info" : result
        }
        if len(result) == 0:
            code = 201
            message = 'no face detect'
            # final_res = {"image_name" : image_name}
            final_res = {}
        else:
            code = 200
            message = 'success'
        return code, message, final_res
if __name__ == '__main__':
    app_name = getattr(PoseHandler, '__name__')
    pose_estimator = PoseEstimator(app_name=app_name)
    deploy(8080,PoseHandler,1,None,pose_estimator)

