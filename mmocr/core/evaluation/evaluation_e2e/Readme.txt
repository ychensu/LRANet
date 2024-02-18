Simply run:
    python main.py

Require (python3):
    pip install Polygon3

Example of one instance of the json result format:
[
    {
     "image_id": 1001, 
     "category_id": 1, 
     "polys": [[x1,y1], [x2, y2], ..., [xn, yn]], 
     "rec": "L164", 
     "score": 0.9123724699020386
    },
]

For instruction of creating custom eval dataset, please refer to 'create_custom_test.py'