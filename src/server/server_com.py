#!/usr/bin/env python
# todo: ImageHandler process raw image data
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
    def __in_post(self,req):
        start = time.time()
        if req.get('image_name') == None or req.get('image_data') == None:
            return_code = 203
            res = {}
            msg = 'image_data or image_name lost'
            return return_code,res,msg

        image = self.build_image(req)
        self.dlog.debug('alg time[%s:np] : %1.4f' % (str(tornado.process.task_id()), time.time() - start))
        if image is None:
            return_code = 202
            # res = {"image_name" : req.get("image_name")}
            res = {}
            msg = 'image_data loading failded'
            return return_code,res,msg
        else:
            try:
                if self.alg_timeout:
                    return_code, res, msg = func_timeout(self.alg_timeout, self.algorithm, args=(req, image))
                else:
                    return_code, res, msg = self.algorithm(req, image)
                return return_code,res,msg
            
            except FunctionTimedOut:
                    return_code = 203
                    res = {"image_name" : req.get("image_name")}
                    msg = 'Algorithm Timeout'
                    self.dlog.debug('Algorithm Timeout')
                    return return_code,res,msg
            except Exception as e:
                return_code = 203
                res = {"image_name" : req.get("image_name")}
                self.dlog.debug(' error: ' + str(e))
                msg = e
                return return_code,res,msg

    @gen.coroutine
    def post(self):
        start = time.time()
        self.set_header("Content-Type", "application/json")
        req = json.loads(self.request.body)
        return_code , res, msg = self.__in_post(req)
        output = json.dumps(self.build_resp(return_code,res,msg))
        self.write_output(output)
        self.dlog.debug('alg time[%s:cv] : %1.4f' % (str(tornado.process.task_id()), time.time() - start))

    @staticmethod
    def build_resp(code,res,message=None):
        resp = {}
        resp['result_code'] = code
        resp['result_data'] = res
        # resp['message'] = message
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
