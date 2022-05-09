
#!/usr/bin/env python
# todo: ImageHandler process raw image data
from tkinter import image_names
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen as gen

import json
import os
import time
import uuid
import base64
import requests
import numpy as np
import cv2
from func_timeout import func_timeout, FunctionTimedOut
from .dlogger import dlogger
#from pudb import set_trace

cur_path = os.path.dirname(__file__)  # current path
__PAGE__ = os.path.join(cur_path, 'index/')  # storage path for demo page


class HealthHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('OK')


class BaseHandler(tornado.web.RequestHandler):
    def initialize(self, app_name, alg_timeout,args):
        self.dlog = dlogger(app_name)
        self.alg_timeout = alg_timeout
        self.args = args
        self.init(args)

    def init(self,args):
        pass

    @staticmethod
    def decode_image(b64):
        npimage = np.fromstring(b64, np.uint8)
        img_restore = cv2.imdecode(npimage, cv2.IMREAD_COLOR)
        return img_restore
    def write_output(self,out):
        self.dlog.debug(out)
        self.write(out)
    @gen.coroutine
    def post(self):
        self.set_header("Content-Type", "application/json")
        start = time.time()
        req = json.loads(self.request.body)
        image = self.build_image(req)
        self.dlog.debug('alg time[%s:np] : %1.4f' % (str(tornado.process.task_id()), time.time() - start))
        if image is None:
            return_code = 202
            res = {"image_name" : req.get("image_name")}
            output = json.dumps(self.build_resp(return_code,res,message='image_data loading failed'))
            self.write_output(output)
        else:
            try:
                if self.alg_timeout:
                    return_code, msg, res = func_timeout(self.alg_timeout, self.algorithm, args=(req, image))
                else:
                    return_code, msg, res = self.algorithm(req, image)
                self.write_output(json.dumps(self.build_resp(return_code, res,message=msg), ensure_ascii=False))
            
            except FunctionTimedOut:
                    return_code = 203
                    res = {"image_name" : req.get("image_name")}
                    self.dlog.debug('TimeOut Error')
                    self.write_output(json.dumps(self.build_resp(return_code,res,message='Algorithm Timeout')))
            except Exception as e:
                return_code = 203
                res = {"image_name" : req.get("image_name")}
                self.dlog.debug(' error: ' + str(e))
                self.write_output(json.dumps(self.build_resp(return_code,res,message=str(e))))
        
        self.dlog.debug('alg time[%s:cv] : %1.4f' % (str(tornado.process.task_id()), time.time() - start))

    @staticmethod
    def build_resp(code,res,message=None):
        resp = {}
        resp['result_code'] = code
        resp['result_data'] = res
        if code == 200:
            message = 'success'
        elif code == 202:
            message = 'wrong image format'
        else:
            message = message
        resp['message'] = message
        return resp

    def algorithm(self, req, image):
        """
        define you algorithm to do somthing
        """
        try:
            image_name = req.get('image_name')
            ret = "hello tornado!"
            return_code = 200
            return return_code , {"image_name":image_name,"res":ret}
        except:
            return_code = 203
            return return_code , {"image_name":image_name}

    @staticmethod
    def build_image(req):
        try:
            image_b64 = req.get("image_data")
            if image_b64 == None:
                return None
            if not isinstance(image_b64,str):
                return None
            image = base64.b64decode(image_b64)
            npimage = BaseHandler.decode_image(image)
            return npimage
        except:
            return None # img recog failed


def deploy(port, handler, num_processes=1, alg_timeout=None,args=None):

    app_name = getattr(handler, '__name__')
    base = str(getattr(handler, '__bases__')[0])
    use_bs64 = False
    if 'BaseHandler' in base:
        handlers = [
            (r"/health", HealthHandler),
            (r"/faceAngle", handler, dict(app_name=app_name, alg_timeout=alg_timeout,args=args)),
            (r"/(.*)", tornado.web.StaticFileHandler, {'path': __PAGE__})
        ]
        use_bs64 = True
    else:
        handlers = [
            (r"/health", HealthHandler),
            (r"/faceAngle", handler, dict(app_name=app_name, alg_timeout=alg_timeout)),
            (r"/(.*)", tornado.web.StaticFileHandler, {'path': __PAGE__})
        ]
    app = tornado.web.Application(handlers=handlers)
    sockets = tornado.netutil.bind_sockets(port)
    if num_processes > 1:
        tornado.process.fork_processes(num_processes=num_processes)

    http_server = tornado.httpserver.HTTPServer(app)
    http_server.add_sockets(sockets)
    print("webapp:{} start listening port {} use_bs64 {}".format(app_name, port, str(use_bs64)))
    tornado.ioloop.IOLoop.instance().start()
