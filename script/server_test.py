from server.client_com import ImageServerClient
import sys
if __name__ == '__main__':
    img_path = sys.argv[1] 
    d = ImageServerClient('http://127.0.0.1:8080/faceAngle')
    # d = ImageServerClient('http://aiipgateway.jiutian.hq.cmcc/facerec/dev/pose/inference')
    image = open(img_path, 'rb').read()
    idict = d.encode_warpimg(image)
    idict['image_name'] = img_path
    req = d.send(idict)
    print(req)