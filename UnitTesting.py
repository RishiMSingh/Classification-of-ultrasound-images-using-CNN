import os
import unittest
from app import app, predict_model2, load_model

#####  Unit testing ####

##These unit tests have been created to test the functionality of the system.

class FlaskTestCase(unittest.TestCase):
    #Flask set up test

    #Segmented Classification Testing

    model = load_model('ResNet50Segmentation.h5')

    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type = 'html/text')
        self.assertEqual(response.status_code, 200)
        
    def test_classifierS_loads(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type ='html/text')
        self.assertTrue(b'Fetal Ultrasound' in response.data)

    def test_classifierS1_loads(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type ='html/text')
        self.assertTrue(b'CRL image for Segmented classification' in response.data)

    def test_classifierS2_loads(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type ='html/text')
        self.assertTrue(b'More Information' in response.data)

    def test_classifierS3_loads(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type ='html/text')
        self.assertTrue(b'predict' in response.data)

    #Non Segmented Classification. 

    def test_classifierNS(self):
        tester = app.test_client(self)
        response = tester.get('/NonSeg', content_type = 'html/text')
        self.assertEqual(response.status_code, 200)

    def test_classifierNS_loads1(self):
        tester = app.test_client(self)
        response = tester.get('/NonSeg', content_type ='html/text')
        self.assertTrue(b'CRL image for Non Segmented classification' in response.data)

    def test_classifierNS_loads2(self):
        tester = app.test_client(self)
        response = tester.get('/NonSeg', content_type ='html/text')
        self.assertTrue(b'Upload Image' in response.data)

    def test_classifierNS_loads3(self):
        tester = app.test_client(self)
        response = tester.get('/NonSeg', content_type ='html/text')
        self.assertTrue(b'More Information' in response.data)

    def test_classifierNS_loads4(self):
        tester = app.test_client(self)
        response = tester.get('/NonSeg', content_type ='html/text')
        self.assertTrue(b'jpg' in response.data)

    #ResNet Segmentation

    def test_classificationSegmentation1(self):
        result = predict_model2('/Users/RishiSingh/crl-images/dataset/Seg/Segmented-Dataset/Testing/Bad/3 CRL (6).jpg',model = load_model('ResNet50Segmentation.h5'))
        self.assertEquals(result, 'tf.Tensor([[0.03306195]], shape=(1, 1), dtype=float32)')

    def test_classificationSegmentation2(self):
        result = predict_model2('/Users/RishiSingh/crl-images/dataset/Seg/Segmented-Dataset/Testing/Bad/3 CRL (6).jpg',model = load_model('ResNet50Segmentation.h5'))
        print(result)
        self.assertEquals(result, 'array([[-1.926548]], dtype=float32)')

    def test_classificationSegmentation3(self):
        result = predict_model2('/Users/RishiSingh/crl-images/dataset/Seg/Segmented-Dataset/Testing/Bad/3 CRL (15).jpg',model = load_model('ResNet50Segmentation.h5'))
        print(result)
        self.assertEquals(result, array([[-3.3757517]], dtype=float32))

    def test_classificationSegmentation4(self):
        result = predict_model2('/Users/RishiSingh/crl-images/dataset/Seg/Segmented-Dataset/Testing/Bad/3 CRL (20).jpg',model = load_model('ResNet50Segmentation.h5'))
        print(result)
        self.assertEquals(result, array([[-3.3757517]], dtype=float32))

    def test_classificationSegmentation5(self):
        result = predict_model2('/Users/RishiSingh/crl-images/dataset/Seg/Segmented-Dataset/Testing/Bad/3 CRL (13).jpg',model = load_model('ResNet50Segmentation.h5'))
        print(result)
        self.assertEquals(result, array([[-3.3757517]], dtype=float32))


if __name__ == '__main__':
    unittest.main()