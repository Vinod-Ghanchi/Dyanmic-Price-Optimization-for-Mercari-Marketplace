# Dynamic Price Optimization for Mercari Marketplace

## Project Overview

This project focuses on building an advanced predictive model to accurately estimate the selling price of products listed on Mercari, a large community-powered marketplace. By analyzing various product features such as item condition, brand, category, and description, the model aims to provide sellers with optimal pricing strategies. This not only helps sellers set competitive prices to increase sales but also offers insights into how different factors influence product pricing in the e-commerce landscape.

The dynamic nature of e-commerce pricing makes this project particularly valuable. A reliable price prediction tool can lead to more informed decision-making for both buyers and sellers, enhancing market efficiency and improving the overall transaction experience.

## Table of Contents

1.  [Objective](#objective)
2.  [Dataset](#dataset)
3.  [Data Preprocessing](#data-preprocessing)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Modeling & Hypothesis Testing](#modeling--hypothesis-testing)
6.  [Conclusion & Insights](#conclusion--insights)
7.  [Challenges & Future Scope](#challenges--future-scope)
8.  [Team Members](#team-members)

## 1. Objective

The primary objective is to develop a robust predictive model for estimating product selling prices on Mercari. This model will help sellers:
* Determine optimal pricing strategies.
* Set competitive prices to increase sales.
* Gain insights into factors influencing product pricing.

This project aims to improve market efficiency by facilitating fairer pricing and enhancing the overall transaction experience for both buyers and sellers.

## 2. Dataset

The project utilizes the **Mercari Price Suggestion Dataset**, publicly available on Kaggle. This dataset is well-structured and contains a substantial number of entries, including various product attributes essential for comprehensive pricing dynamics analysis.

**Dataset Columns:**
* `train_id`: Unique ID for the training item.
* `name`: Name of the product.
* `item_condition_id`: Condition of the item (e.g., new, used).
* `category_name`: Hierarchical category of the product (e.g., `Men/Tops/T-shirts`).
* `brand_name`: Brand of the product.
* `price`: The target variable; the selling price of the product.
* `shipping`: Indicates if shipping is paid by the seller (0) or buyer (1).
* `item_description`: Detailed description of the item.

**Example Data:**

| train\_id | name                          | item\_condition\_id | category\_name                                    | brand\_name | price | shipping | item\_description                                |
| :-------- | :---------------------------- | :------------------ | :------------------------------------------------ | :---------- | :---- | :------- | :----------------------------------------------- |
| 0         | MLB Cincinnati Reds T Shirt XL | 3                   | Men/Tops/T-shirts                                 | NaN         | 10.0  | 1        | No description yet                               |
| 1         | Razer BlackWidow Chroma Keyboard | 3                   | Electronics/Computers & Tablets/Components & P... | Razer       | 52.0  | 0        | This keyboard is in great condition and works ... |

## 3. Data Preprocessing

Textual features (`name`, `item_description`, `category_name`, `brand_name`) were standardized and cleaned.

**Key Preprocessing Steps:**
* **Missing Values:** Replaced missing values in `brand_name`, `category_name`, and `item_description` with "Null" or "missing" as appropriate.
* **Text Normalization:** Converted text to lowercase, stripped unnecessary spaces, and processed special characters.
* **Category Splitting:** Hierarchical categories (e.g., `Men/Tops/T-shirts`) were split into three levels: `general_cat`, `sub_cat1`, and `sub_cat2` for granular analysis.
* **Numerical Feature Transformation:** `price` was log-transformed to address skewness, and all numerical features were standardized for model consistency.

## 4. Exploratory Data Analysis (EDA)

EDA was crucial for identifying key features and relationships influencing product pricing.

**Key Learnings from EDA:**

* **Price Distribution:** The price distribution was heavily skewed towards lower-priced items, necessitating log transformation for modeling.
    * **Image: Price Distribution Before and After Log Transformation**
        ![image](https://github.com/user-attachments/assets/35ae4dda-fb81-4de7-b733-22f8caed7953)


* **Category Analysis:**
    * The dataset contains `1,288` distinct categories.
    * "Women/Athletics Pants Tights Leggings" is the most frequent product category, indicating its dominance.
    * **Image: Top 10 Most Frequent Categories**
        ![image](https://github.com/user-attachments/assets/aaae1e8f-4707-41db-8087-a33ded213457)

    * Categories like "Electronics" and "Vintage & Collectibles" exhibit a wide price range, while "Beauty" and "Handmade" show more consistent pricing with fewer outliers.
    * **Image: Price Distribution by General Category (Box Plot)**
        ![image](https://github.com/user-attachments/assets/f7f0d460-e92e-4876-9636-b872ce7c8ff8)

    * The top 10 subcategories are dominated by popular items like athletic apparel and makeup, while the bottom 10 are niche.
        * **Image: Top 10 and Bottom 10 Subcategories**
            ![image](https://github.com/user-attachments/assets/0c7b589b-c2bd-481f-a080-1e6c1317859c)


* **Shipping Responsibility:** Buyers tend to pay shipping fees more frequently for higher-priced items.
    * **Image: Price Distribution by Shipping Responsibility**
        ![image](https://github.com/user-attachments/assets/8bccc289-6752-4c82-9dcb-60d1a9ea5af9)


* **Description Length vs. Price:** A general trend suggests that longer descriptions tend to correspond to higher prices, though the relationship is not strictly linear.
    * **Image: Description Length vs. Price (Scatter Plot)**
        ![image](https://github.com/user-attachments/assets/86c066c1-8add-4ea3-8b1e-ce8f67954b8c)


## 5. Modeling & Hypothesis Testing

We evaluated several regression models to predict product prices, starting with simpler baselines and progressing to more advanced techniques.

**Baseline Models:**
* Linear Regression
* Ridge Regression
* Decision Tree Regressor

| Model                 | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R^2   | Explained Variance |
| :-------------------- | :------------------------ | :----------------------- | :---- | :----------------- |
| LinearRegression      | 0.56                      | 0.52                     | 0.07  | 0.07               |
| Ridge                 | 0.56                      | 0.52                     | 0.07  | 0.07               |
| DecisionTreeRegressor | 0.56                      | 0.59                     | -0.05 | -0.05              |

*Initial models showed limited predictive power, with Decision Tree Regressor performing poorly before tuning.*

**Hyperparameter Tuning (Decision Tree Regressor):**
Manual tuning of `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `max_leaf_nodes` significantly improved performance.

| Metric             | Value    |
| :----------------- | :------- |
| Mean Absolute Error | 0.501    |
| Mean Squared Error | 0.427    |
| R^2                 | 0.243    |
| Explained Variance | 0.243    |

*Hyperparameter tuning reduced MAE from 0.56 to 0.50 and improved R² from -0.05 to 0.24, demonstrating enhanced predictive accuracy and generalization.*

**Advanced Model: LightGBM**
LightGBM, a powerful gradient boosting framework, was chosen for its efficiency and scalability with large datasets, and its ability to capture non-linear relationships.

| Metric             | Value    |
| :----------------- | :------- |
| Mean Absolute Error | 0.468    |
| Mean Squared Error | 0.372    |
| R^2                 | 0.340    |
| Explained Variance | 0.340    |

*LightGBM achieved the best performance with an MAE of 0.47 and R² of 0.34, significantly outperforming other models and effectively capturing complex data patterns.*

**Model Comparison (Learning Curves):**
Learning curves were plotted to visualize model performance with increasing training data size, helping to identify overfitting or underfitting.

* **Image: Learning Curves for All Models**
    ![image](https://github.com/user-attachments/assets/d4cd8d30-d048-42c9-a114-321838661264)


## 6. Conclusion & Insights

This project successfully developed an advanced predictive model for Mercari product pricing. Through extensive EDA, we identified crucial features and relationships influencing prices. While simpler models like Linear Regression and Ridge provided a baseline, hyperparameter tuning significantly boosted the Decision Tree model's performance. Ultimately, LightGBM emerged as the superior model, demonstrating its effectiveness in handling large datasets and complex, non-linear relationships, and achieving the highest R² score.

## 7. Challenges & Future Scope

**Challenges Faced:**
* **Category Data Complexity:** The `category_name` field had nested structures (e.g., `Electronics/Computers/Tablets`), requiring careful splitting into subcategories and cleaning.
* **Handling Large Data:** The dataset size caused computational bottlenecks, making model training and hyperparameter tuning resource-intensive.
* **Data Cleaning:** Issues like missing `brand_name`, inconsistent prices, and outliers required careful handling to ensure data quality without degrading model performance.
* **Hyperparameter Tuning:** Tuning improved performance but was computationally expensive, requiring simplified approaches to balance accuracy and efficiency.

**Future Work:**
* **Extensive Testing on Unseen Data:** Additional cross-validation and stress-testing are needed to ensure model robustness and generalization.
* **Cross-Domain Analysis:** Exploring external factors like market trends or seasonal variations could enhance predictive accuracy.
* **Advanced NLP Techniques:** Utilizing advanced NLP models (e.g., BERT or GPT) could improve predictions by capturing nuances in text-based features.

## 8. Team Members

* Vinod Ghanchi
* Ashay Katre
* Yashvi Mehta
