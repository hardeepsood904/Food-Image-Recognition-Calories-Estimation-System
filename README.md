# Food-Image-Recognition-Calories-Estimation-System

Workflow...
Workflow of Food Image Recognition and Calorie Estimation System
Step 1: Data Collection

The dataset was collected from Kaggle, containing:

Food images for various categories (e.g., pizza, burger, pasta, etc.)

An Excel file that includes calorie information and health benefits for each food item.

This dataset serves as the foundation for both training the deep learning model and retrieving calorie values.

Step 2: Data Preprocessing

All the food images were preprocessed to ensure consistency and quality before training.

Preprocessing steps include:

Image resizing to a uniform size (e.g., 224×224 pixels).

Normalization of pixel values (scaling between 0–1).

Label encoding — converting food names into numerical form.

Splitting the dataset into:

Training set

Validation set

Testing set

Data augmentation (optional) for better generalization — e.g., rotation, flipping, zooming.

Step 3: Loading and Linking Calorie Data

The Excel file containing calorie and health benefit information was loaded using Pandas.

Each food item in the dataset is linked to its calorie and benefit details.

Example:

Food Name	Calories (kcal)	Health Benefits
Pizza	285	Source of protein and calcium
Salad	150	High in fiber, good for digestion
Step 4: Model Building (Deep Learning Model)

A deep learning model was developed for food image recognition.

Convolutional Neural Network (CNN) was used.

The model learns to extract features from images and classify them into corresponding food categories.

Process:

Define CNN architecture model.

Compile the model with optimizer, loss function, and metrics.

Train the model using the preprocessed training data.

Validate the model using the validation data.

Step 5: Model Training

The dataset was trained on the model with appropriate epochs and batch size.

During training:

Training accuracy and loss were monitored.

Validation accuracy and loss were used to prevent overfitting.

After training, the best-performing model was saved for testing.

Step 6: Model Testing

In the testing phase:

The trained model is loaded.

A test image is uploaded from the test dataset.

The model predicts the food name based on the image.

Example Output:

Predicted Food: Pizza

Step 7: Calorie and Benefit Retrieval

Once the food name is recognized:

The system fetches the corresponding calorie value and health benefits from the Excel file.

Example:

Food: Pizza
Calories: 285 kcal
Benefits: Good source of protein and calcium


This step combines machine learning output with stored nutritional data for a complete prediction.

Step 8: Output Display

The final output is displayed to the user, showing:

Predicted food name

Estimated calories

Nutritional or health benefits

Example:

🍕 Food Name: Pizza
🔥 Calories: 285 kcal
💪 Benefits: High in protein and calcium

Step 9: Evaluation

The model’s performance was evaluated using:

Accuracy — for classification of food images.

Loss curve — to monitor learning behavior.

Optional: Confusion matrix to visualize classification performance.

Step 10: Future Enhancement

Integrate portion or volume estimation for more accurate calorie calculation.

Expand dataset with more diverse food types.

Deploy the system as a web or mobile application for real-world use.

🧩 Workflow Summary Diagram
        ┌─────────────────────────────┐
        │ 1. Data Collection (Kaggle) │
        └────────────┬────────────────┘
                     ↓
        ┌───────────────────────────┐
        │ 2. Data Preprocessing     │
        │ (Resize, Normalize, Split)│
        └────────────┬──────────────┘
                     ↓
        ┌──────────────────────────┐
        │ 3. Load Excel (Calories) │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ 4. Build CNN Model       │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ 5. Train the Model       │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ 6. Test Image Prediction │
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │ 7. Fetch Calories & Info │
        └────────────┬─────────────┘
                     ↓
        ┌─────────────────────────────┐
        │ 8. Display Final Output     │
        │ (Food + Calories + Benefit) │
        └─────────────────────────────┘
