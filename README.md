# Fake Currency Note Detection

## **Overview**
Fake currency detection is a critical task for financial institutions and businesses. This project utilizes **Convolutional Neural Networks (CNNs)** to build an image-based deep learning model capable of distinguishing between real and fake currency notes.


## **Features**
- **Image-based classification**: Detects fake notes using deep learning techniques.
- **Automated analysis**: No manual intervention required.
- **Scalable and adaptable**: Can be trained on multiple currencies.

## **Technology Stack**
- **Programming Language**: Python
- **Framework**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Dataset Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn


## **Model Architecture**
1. **Input Layer** – Accepts currency note images.
2. **Convolutional Layers** – Extracts key features such as edges and textures.
3. **Pooling Layers** – Reduces dimensionality while retaining essential information.
4. **Fully Connected Layers** – Classifies the image as real or fake.
5. **Output Layer** – Uses Softmax activation to predict probabilities.


## **Usage**
### **1. Train the Model**
```sh
python train.py
```

### **2. Test the Model**
```sh
python test.py --image path/to/image.jpg
```

### **3. Run the Web Application**
```sh
python app.py
```

## **Evaluation Metrics**
| Metric  | Value |
|---------|-------|
| Accuracy | 95%  |
| Precision | 94% |
| Recall | 96% |
| F1-Score | 95% |

## **Future Enhancements**
- Train the model on multiple currencies (USD, INR, EUR, etc.).
- Integrate **OCR (Optical Character Recognition)** for serial number verification.
- Deploy the model on **mobile devices** using TensorFlow Lite.


