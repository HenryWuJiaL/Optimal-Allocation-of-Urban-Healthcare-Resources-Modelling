{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.045603,
     "end_time": "2021-01-01T17:52:39.896559",
     "exception": false,
     "start_time": "2021-01-01T17:52:39.850956",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**Insurance companies can't predict the future, so how do they come up with such quote?**\n",
    "\n",
    "By understanding the relationship between the demographics of primary beneficiaries (age, sex, BMI, number of children, smoker and location) and their medical costs, insurance companies will then (i) compare you to their data, (ii) predict your medical costs, and (iii) calculate a corresponding insurance quote. \n",
    "\n",
    "Let's take a look at the first two steps (grouping you and predicting medical resource) with the following example:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:39.993955Z",
     "iopub.status.busy": "2021-01-01T17:52:39.993191Z",
     "iopub.status.idle": "2021-01-01T17:52:41.253674Z",
     "shell.execute_reply": "2021-01-01T17:52:41.254351Z"
    },
    "papermill": {
     "duration": 1.312813,
     "end_time": "2021-01-01T17:52:41.254568",
     "exception": false,
     "start_time": "2021-01-01T17:52:39.941755",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import scipy.stats as stats\n",
    "%matplotlib inline\n",
    "params={\"figure.facecolor\":(0.0,0.0,0.0,0),\n",
    "        \"axes.facecolor\":(1.0,1.0,1.0,1.0),\n",
    "        \"savefig.facecolor\":(0.0,0.0,0.0,0)}\n",
    "plt.rcParams.update(params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
    "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:41.352875Z",
     "iopub.status.busy": "2021-01-01T17:52:41.352146Z",
     "iopub.status.idle": "2021-01-01T17:52:41.401113Z",
     "shell.execute_reply": "2021-01-01T17:52:41.401662Z"
    },
    "papermill": {
     "duration": 0.097198,
     "end_time": "2021-01-01T17:52:41.401825",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.304627",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>BMI</th>\n",
       "      <th>Children</th>\n",
       "      <th>Smoker</th>\n",
       "      <th>Region</th>\n",
       "      <th>Charges</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>19</td>\n",
       "      <td>female</td>\n",
       "      <td>27.900</td>\n",
       "      <td>0</td>\n",
       "      <td>yes</td>\n",
       "      <td>southwest</td>\n",
       "      <td>16884.92400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18</td>\n",
       "      <td>male</td>\n",
       "      <td>33.770</td>\n",
       "      <td>1</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>1725.55230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>male</td>\n",
       "      <td>33.000</td>\n",
       "      <td>3</td>\n",
       "      <td>no</td>\n",
       "      <td>southeast</td>\n",
       "      <td>4449.46200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>33</td>\n",
       "      <td>male</td>\n",
       "      <td>22.705</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>21984.47061</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>32</td>\n",
       "      <td>male</td>\n",
       "      <td>28.880</td>\n",
       "      <td>0</td>\n",
       "      <td>no</td>\n",
       "      <td>northwest</td>\n",
       "      <td>3866.85520</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age     Sex     BMI  Children Smoker     Region      Charges\n",
       "0   19  female  27.900         0    yes  southwest  16884.92400\n",
       "1   18    male  33.770         1     no  southeast   1725.55230\n",
       "2   28    male  33.000         3     no  southeast   4449.46200\n",
       "3   33    male  22.705         0     no  northwest  21984.47061\n",
       "4   32    male  28.880         0     no  northwest   3866.85520"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv(\"../input/insurance/insurance.csv\")\n",
    "df.columns=df.columns.to_series().apply(lambda x:x.title() if x!=\"bmi\" else x.upper())\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:41.502158Z",
     "iopub.status.busy": "2021-01-01T17:52:41.501088Z",
     "iopub.status.idle": "2021-01-01T17:52:41.506459Z",
     "shell.execute_reply": "2021-01-01T17:52:41.505633Z"
    },
    "papermill": {
     "duration": 0.058757,
     "end_time": "2021-01-01T17:52:41.506582",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.447825",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1338, 7)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.045951,
     "end_time": "2021-01-01T17:52:41.600082",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.554131",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The dataset has 1338 observations/individuals which can be described in the following seven features:\n",
    "\n",
    "* *Age*: the age of primary beneficiary\n",
    "\n",
    "* *Sex*: female or male\n",
    "\n",
    "* *BMI*: Body Mass Index (kg/m^2), a BMI between 18.5 kg/m^2 to 24.9 kg/m^2 indicates a healthy individual while anything less indicates underweight and more indicates overweight\n",
    "\n",
    "* *Children*: the number of children/dependents covered by insurance\n",
    "\n",
    "* *Smoker*: if the primary beneficiary smokes\n",
    "\n",
    "* *Region*: the beneficiary's residential area in the US (northeast, southeast, southwest, northwest)\n",
    "\n",
    "* *Charges*: medical costs billed by the insurance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:41.715569Z",
     "iopub.status.busy": "2021-01-01T17:52:41.714237Z",
     "iopub.status.idle": "2021-01-01T17:52:41.718783Z",
     "shell.execute_reply": "2021-01-01T17:52:41.719537Z"
    },
    "papermill": {
     "duration": 0.073866,
     "end_time": "2021-01-01T17:52:41.719714",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.645848",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1338 entries, 0 to 1337\n",
      "Data columns (total 7 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   Age       1338 non-null   int64  \n",
      " 1   Sex       1338 non-null   object \n",
      " 2   BMI       1338 non-null   float64\n",
      " 3   Children  1338 non-null   int64  \n",
      " 4   Smoker    1338 non-null   object \n",
      " 5   Region    1338 non-null   object \n",
      " 6   Charges   1338 non-null   float64\n",
      "dtypes: float64(2), int64(2), object(3)\n",
      "memory usage: 73.3+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.045846,
     "end_time": "2021-01-01T17:52:41.815132",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.769286",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "There are no null values, but there is a mixture of categorical and numerical features (the categorical features will later be changed into numerical format)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.04881,
     "end_time": "2021-01-01T17:52:41.913354",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.864544",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Let's further examine each feature:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.046641,
     "end_time": "2021-01-01T17:52:42.008153",
     "exception": false,
     "start_time": "2021-01-01T17:52:41.961512",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#### *Age*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:42.115543Z",
     "iopub.status.busy": "2021-01-01T17:52:42.111574Z",
     "iopub.status.idle": "2021-01-01T17:52:42.492184Z",
     "shell.execute_reply": "2021-01-01T17:52:42.491224Z"
    },
    "papermill": {
     "duration": 0.436092,
     "end_time": "2021-01-01T17:52:42.492534",
     "exception": false,
     "start_time": "2021-01-01T17:52:42.056442",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x7fd4c16af290>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x=df[\"Age\"],y=df[\"Charges\"],hue=df[\"Smoker\"],palette={\"yes\":\"salmon\",\"no\":\"skyblue\"})\n",
    "plt.legend(bbox_to_anchor=(1,0.5),loc=6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.054298,
     "end_time": "2021-01-01T17:52:42.607356",
     "exception": false,
     "start_time": "2021-01-01T17:52:42.553058",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* There is a slight positive correlation between the age and charges.\n",
    "* The data can also be separated into three groupings (low, average and high charges) depending on if the individual was a smoker."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.067916,
     "end_time": "2021-01-01T17:52:42.724397",
     "exception": false,
     "start_time": "2021-01-01T17:52:42.656481",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#### *Sex*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:42.860882Z",
     "iopub.status.busy": "2021-01-01T17:52:42.859757Z",
     "iopub.status.idle": "2021-01-01T17:52:43.039193Z",
     "shell.execute_reply": "2021-01-01T17:52:43.040242Z"
    },
    "papermill": {
     "duration": 0.254049,
     "end_time": "2021-01-01T17:52:43.040529",
     "exception": false,
     "start_time": "2021-01-01T17:52:42.786480",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4c03daa10>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=df[\"Sex\"],y=df[\"Charges\"],palette={\"female\":\"lemonchiffon\",\"male\":\"lightgreen\"})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.056741,
     "end_time": "2021-01-01T17:52:43.178527",
     "exception": false,
     "start_time": "2021-01-01T17:52:43.121786",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* Including the \"outliers\", both females and males have similar range.\n",
    "* Excluding the \"outliers\", males have higher charges."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.054573,
     "end_time": "2021-01-01T17:52:43.290523",
     "exception": false,
     "start_time": "2021-01-01T17:52:43.235950",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#### *BMI*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:43.403888Z",
     "iopub.status.busy": "2021-01-01T17:52:43.402806Z",
     "iopub.status.idle": "2021-01-01T17:52:43.698634Z",
     "shell.execute_reply": "2021-01-01T17:52:43.699259Z"
    },
    "papermill": {
     "duration": 0.354007,
     "end_time": "2021-01-01T17:52:43.699456",
     "exception": false,
     "start_time": "2021-01-01T17:52:43.345449",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x7fd4c02eed10>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.scatterplot(x=df[\"BMI\"],y=df[\"Charges\"],hue=df[\"Smoker\"],palette={\"yes\":\"salmon\",\"no\":\"skyblue\"})\n",
    "plt.legend(bbox_to_anchor=(1,0.5),loc=6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.052385,
     "end_time": "2021-01-01T17:52:43.812632",
     "exception": false,
     "start_time": "2021-01-01T17:52:43.760247",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* Charges do not necessarily increase with BMI, it depends on if the individual smokes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.0518,
     "end_time": "2021-01-01T17:52:43.916636",
     "exception": false,
     "start_time": "2021-01-01T17:52:43.864836",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#### Number of *Children*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:44.031495Z",
     "iopub.status.busy": "2021-01-01T17:52:44.030271Z",
     "iopub.status.idle": "2021-01-01T17:52:44.270149Z",
     "shell.execute_reply": "2021-01-01T17:52:44.270796Z"
    },
    "papermill": {
     "duration": 0.30018,
     "end_time": "2021-01-01T17:52:44.270961",
     "exception": false,
     "start_time": "2021-01-01T17:52:43.970781",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4c030ac10>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=df[\"Children\"],y=df[\"Charges\"],palette=\"Set2\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.054913,
     "end_time": "2021-01-01T17:52:44.385178",
     "exception": false,
     "start_time": "2021-01-01T17:52:44.330265",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* Having more children may be more expensive than those with no or less children."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.055737,
     "end_time": "2021-01-01T17:52:44.494490",
     "exception": false,
     "start_time": "2021-01-01T17:52:44.438753",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#### *Smoker* Status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:44.613260Z",
     "iopub.status.busy": "2021-01-01T17:52:44.612572Z",
     "iopub.status.idle": "2021-01-01T17:52:44.873641Z",
     "shell.execute_reply": "2021-01-01T17:52:44.872815Z"
    },
    "papermill": {
     "duration": 0.322901,
     "end_time": "2021-01-01T17:52:44.873785",
     "exception": false,
     "start_time": "2021-01-01T17:52:44.550884",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4c01d79d0>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=df[\"Smoker\"],y=df[\"Charges\"],palette={\"yes\":\"salmon\",\"no\":\"skyblue\"})"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.058235,
     "end_time": "2021-01-01T17:52:44.991068",
     "exception": false,
     "start_time": "2021-01-01T17:52:44.932833",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* Smokers have a chargers than non-smokers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.061503,
     "end_time": "2021-01-01T17:52:45.112190",
     "exception": false,
     "start_time": "2021-01-01T17:52:45.050687",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "#### *Region*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:45.248965Z",
     "iopub.status.busy": "2021-01-01T17:52:45.244765Z",
     "iopub.status.idle": "2021-01-01T17:52:45.450182Z",
     "shell.execute_reply": "2021-01-01T17:52:45.448982Z"
    },
    "papermill": {
     "duration": 0.275454,
     "end_time": "2021-01-01T17:52:45.450457",
     "exception": false,
     "start_time": "2021-01-01T17:52:45.175003",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4c01ea310>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=df[\"Region\"],y=df[\"Charges\"],palette=\"Set3\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.060057,
     "end_time": "2021-01-01T17:52:45.571303",
     "exception": false,
     "start_time": "2021-01-01T17:52:45.511246",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* On average there is no obvious relationship between the region and charges, but those in the southeast have a larger spread of higher charges."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.062255,
     "end_time": "2021-01-01T17:52:45.690526",
     "exception": false,
     "start_time": "2021-01-01T17:52:45.628271",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Let us now change the categorical features into numerical format:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:45.815487Z",
     "iopub.status.busy": "2021-01-01T17:52:45.813038Z",
     "iopub.status.idle": "2021-01-01T17:52:45.818754Z",
     "shell.execute_reply": "2021-01-01T17:52:45.818099Z"
    },
    "papermill": {
     "duration": 0.069033,
     "end_time": "2021-01-01T17:52:45.818878",
     "exception": false,
     "start_time": "2021-01-01T17:52:45.749845",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df[\"Sex\"]=pd.get_dummies(df[\"Sex\"],drop_first=True)\n",
    "#where 0=female and 1=male"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:45.943342Z",
     "iopub.status.busy": "2021-01-01T17:52:45.942662Z",
     "iopub.status.idle": "2021-01-01T17:52:45.946169Z",
     "shell.execute_reply": "2021-01-01T17:52:45.945489Z"
    },
    "papermill": {
     "duration": 0.069168,
     "end_time": "2021-01-01T17:52:45.946286",
     "exception": false,
     "start_time": "2021-01-01T17:52:45.877118",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df[\"Smoker\"]=pd.get_dummies(df[\"Smoker\"],drop_first=True)\n",
    "#where 0=non-smoker and 1=smoker"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:46.075307Z",
     "iopub.status.busy": "2021-01-01T17:52:46.074034Z",
     "iopub.status.idle": "2021-01-01T17:52:46.078914Z",
     "shell.execute_reply": "2021-01-01T17:52:46.078220Z"
    },
    "papermill": {
     "duration": 0.075699,
     "end_time": "2021-01-01T17:52:46.079032",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.003333",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_regions=pd.get_dummies(df[\"Region\"])\n",
    "df=df.join(df_regions)\n",
    "df.drop([\"Region\"],axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.056083,
     "end_time": "2021-01-01T17:52:46.191327",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.135244",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The dataset now looks like this:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:46.317849Z",
     "iopub.status.busy": "2021-01-01T17:52:46.317122Z",
     "iopub.status.idle": "2021-01-01T17:52:46.322978Z",
     "shell.execute_reply": "2021-01-01T17:52:46.322287Z"
    },
    "papermill": {
     "duration": 0.075691,
     "end_time": "2021-01-01T17:52:46.323098",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.247407",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>BMI</th>\n",
       "      <th>Children</th>\n",
       "      <th>Smoker</th>\n",
       "      <th>Charges</th>\n",
       "      <th>northeast</th>\n",
       "      <th>northwest</th>\n",
       "      <th>southeast</th>\n",
       "      <th>southwest</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>19</td>\n",
       "      <td>0</td>\n",
       "      <td>27.900</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>16884.92400</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>18</td>\n",
       "      <td>1</td>\n",
       "      <td>33.770</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1725.55230</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>33.000</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>4449.46200</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>33</td>\n",
       "      <td>1</td>\n",
       "      <td>22.705</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>21984.47061</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "      <td>28.880</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3866.85520</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Sex     BMI  Children  Smoker      Charges  northeast  northwest  \\\n",
       "0   19    0  27.900         0       1  16884.92400          0          0   \n",
       "1   18    1  33.770         1       0   1725.55230          0          0   \n",
       "2   28    1  33.000         3       0   4449.46200          0          0   \n",
       "3   33    1  22.705         0       0  21984.47061          0          1   \n",
       "4   32    1  28.880         0       0   3866.85520          0          1   \n",
       "\n",
       "   southeast  southwest  \n",
       "0          0          1  \n",
       "1          1          0  \n",
       "2          1          0  \n",
       "3          0          0  \n",
       "4          0          0  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.05682,
     "end_time": "2021-01-01T17:52:46.438060",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.381240",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "We shall continue exploring our target variable, *Charges*:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:46.578398Z",
     "iopub.status.busy": "2021-01-01T17:52:46.573674Z",
     "iopub.status.idle": "2021-01-01T17:52:46.795808Z",
     "shell.execute_reply": "2021-01-01T17:52:46.795077Z"
    },
    "papermill": {
     "duration": 0.299026,
     "end_time": "2021-01-01T17:52:46.795938",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.496912",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4c00b2490>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(a=df[\"Charges\"],color=\"plum\",hist_kws=dict(edgecolor=\"black\",linewidth=2),bins=20,kde=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:46.925175Z",
     "iopub.status.busy": "2021-01-01T17:52:46.924339Z",
     "iopub.status.idle": "2021-01-01T17:52:46.929991Z",
     "shell.execute_reply": "2021-01-01T17:52:46.929283Z"
    },
    "papermill": {
     "duration": 0.071995,
     "end_time": "2021-01-01T17:52:46.930110",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.858115",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Charges have a skewness of 1.51.\n"
     ]
    }
   ],
   "source": [
    "print(\"Charges have a skewness of {:.2f}.\".format(stats.skew(df[\"Charges\"])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:47.071977Z",
     "iopub.status.busy": "2021-01-01T17:52:47.070551Z",
     "iopub.status.idle": "2021-01-01T17:52:47.075679Z",
     "shell.execute_reply": "2021-01-01T17:52:47.074847Z"
    },
    "papermill": {
     "duration": 0.079685,
     "end_time": "2021-01-01T17:52:47.075820",
     "exception": false,
     "start_time": "2021-01-01T17:52:46.996135",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     1338.000000\n",
       "mean     13270.422265\n",
       "std      12110.011237\n",
       "min       1121.873900\n",
       "25%       4740.287150\n",
       "50%       9382.033000\n",
       "75%      16639.912515\n",
       "max      63770.428010\n",
       "Name: Charges, dtype: float64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"Charges\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:47.266332Z",
     "iopub.status.busy": "2021-01-01T17:52:47.265586Z",
     "iopub.status.idle": "2021-01-01T17:52:47.396294Z",
     "shell.execute_reply": "2021-01-01T17:52:47.395678Z"
    },
    "papermill": {
     "duration": 0.207557,
     "end_time": "2021-01-01T17:52:47.396454",
     "exception": false,
     "start_time": "2021-01-01T17:52:47.188897",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4b978e610>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAq8AAAEGCAYAAABLmnwmAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAWzklEQVR4nO3dfYyd1Z0f8O+ZGY9tTFwWzAy2WYxThWhp2pKADYg2u5uNt7thU0GC8gJ+wSJL2iRVVpVaJV1pRf6p1K5ararIJcA2KQ7bhdLEXZlGxNo0baWC33bJGpsAa5ck2MR4FvAYY7CZefrH3DuZsWfs8dvcOfD5SFdz73lezu+e33D95c5zZ0rTNAEAgBp0dboAAACYKuEVAIBqCK8AAFRDeAUAoBrCKwAA1eg5nZ0XLFjQXHnlleepFAAASLZv3z7QNM2lE207rfB65ZVXZtu2beemKgAAmEAp5SeTbXPZAAAA1RBeAQCohvAKAEA1hFcAAKohvAIAUA3hFQCAagivAABUQ3gFAKAawisAANUQXgEAqIbwCgBANYRXAACqIbwCAFAN4RUAgGoIrwAAVEN4BQCgGsIrAADVEF4BAKiG8AoAQDV6Ol1ATTZs2JB9+/Z1uozTMjAwkCRZsGBBhyuZmRYtWpRbbrml02UAAFMkvJ6Gffv25Wcv/CwXX3Bxp0uZsjfeeCNJcnj4cIcrmXleeeOVTpcAAJwm4fU0XXzBxbn5V27udBlT9tgzjyVJVTVPl/baAAD1cM0rAADVEF4BAKiG8AoAQDWEVwAAqiG8AgBQDeEVAIBqCK8AAFRDeAUAoBrCKwAA1RBeAQCohvAKAEA1hFcAAKohvAIAUA3hFQCAagivAABUQ3gFAKAawisAANUQXgEAqIbwCgBANYRXAACqIbwCAFAN4RUAgGoIrwAAVEN4BQCgGsIrAADVEF4BAKiG8AoAQDWEVwAAqiG8AgBQDeEVAIBqCK8AAFRDeAUAoBrCKwAA1RBeAQCohvAKAEA1hFcAAKohvAIAUA3hFQCAagivAABUQ3gFAKAawisAANWY8eF1w4YN2bBhQ6fLAOg4r4cASU+nCziVffv2dboEgBnB6yFABe+8AgBAm/AKAEA1hFcAAKohvAIAUA3hFQCAagivAABUQ3gFAKAawisAANUQXgEAqIbwCgBANYRXAACqIbwCAFAN4RUAgGoIrwAAVEN4BQCgGsIrAADVEF4BAKiG8AoAQDWEVwAAqiG8AgBQDeEVAIBqCK8AAFRDeAUAoBrCKwAA1RBeAQCohvAKAEA1hFcAAKohvAIAUA3hFQCAagivAABUQ3gFAKAawisAANUQXgEAqIbwCgBANYRXAACqIbwCAFAN4RUAgGoIrwAAVKOn0wUAcHr27t2br3/96xkeHk5XV1fWrl2bjRs3ZmBgIF/60pdy4YUX5oEHHhh9vGjRogwODuYb3/hGfv7znydJ+vr6Mnfu3Nx222357ne/m1tvvTUPPfRQ9u/fn4997GN57LHHsnDhwnz84x/Pgw8+mDVr1mTjxo05cOBA+vr6ctdddyVJHnjggRw4cCAXXXRRDh48mE996lN5+OGHk2RcXXfeeWc2bdqUVatW5aWXXsp9992XJOnp6cnFF1+c1157LZdcckm6urpGx++8884kyfr163PNNdfkO9/5Ti655JLMmzcva9euzfz580fXY926dfniF7+YQ4cO5f77709XV1e6urrS398/Wuv69etz66235tFHH81bb72VV199dXS91q9fn1WrViVJvvnNb2ZoaChHjx7NwMBAPv/5z+eCCy7IunXrctttt+WRRx5J0zSj554/f34GBwdHz9Gua2xtd955Z77//e/nox/9aL71rW+NO769jgMDA1m7dm2+973vpWmarF27dnStZs2alb6+vnzuc5/L/PnzTzhve97290aSfPrTn84jjzySSy+9dLTOtsnqnaozOf5s55wpc5yvGmbi+syE9ZxI9z333DPlne+777577r777vNXzQS2bt2aJFm2bNm0zjuRrVu35tjhY7nq0qs6XcqUPT/wfJJUVfN0eX7g+fTO650R31swFe3Xw+3bt2dwcDDDw8MZGhrKrl278sorr2RoaCh79uzJq6++ml27do0+vummm7Jx48Y888wzo+c6fPhwDh48mD179mTv3r3ZvXt39u/fnyR5/vmR143XX389u3btyptvvjlujsHBwRw7dix79uwZnefw4cMZGhrKzp078/bbb59Q165duzIwMJCjR4/m8ccfz9tvv50kGR4eHj329ddfz6FDh3Lo0KEcPHhwdI4dO3aM1n7kyJHRbVdffXWS5N577x19Llu3bs2xY8fSNE2Gh4fH1bpjx47s3r07e/fuzeuv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\n",
      "text/plain": [
       "<Figure size 864x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(12,4))\n",
    "sns.boxplot(df[\"Charges\"],color=\"plum\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:47.549520Z",
     "iopub.status.busy": "2021-01-01T17:52:47.548100Z",
     "iopub.status.idle": "2021-01-01T17:52:47.850323Z",
     "shell.execute_reply": "2021-01-01T17:52:47.850831Z"
    },
    "papermill": {
     "duration": 0.381237,
     "end_time": "2021-01-01T17:52:47.850993",
     "exception": false,
     "start_time": "2021-01-01T17:52:47.469756",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fd4b9773410>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "mask=np.zeros_like(df.corr(),dtype=np.bool)\n",
    "mask[np.triu_indices_from(mask)]=True\n",
    "sns.heatmap(df.corr(),annot=False,mask=mask,square=True,cmap=\"PuRd\",linewidths=1,linecolor=\"white\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:47.993521Z",
     "iopub.status.busy": "2021-01-01T17:52:47.978873Z",
     "iopub.status.idle": "2021-01-01T17:52:48.000704Z",
     "shell.execute_reply": "2021-01-01T17:52:48.000023Z"
    },
    "papermill": {
     "duration": 0.088612,
     "end_time": "2021-01-01T17:52:48.000834",
     "exception": false,
     "start_time": "2021-01-01T17:52:47.912222",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>BMI</th>\n",
       "      <th>Children</th>\n",
       "      <th>Smoker</th>\n",
       "      <th>Charges</th>\n",
       "      <th>northeast</th>\n",
       "      <th>northwest</th>\n",
       "      <th>southeast</th>\n",
       "      <th>southwest</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.020856</td>\n",
       "      <td>0.109272</td>\n",
       "      <td>0.042469</td>\n",
       "      <td>-0.025019</td>\n",
       "      <td>0.299008</td>\n",
       "      <td>0.002475</td>\n",
       "      <td>-0.000407</td>\n",
       "      <td>-0.011642</td>\n",
       "      <td>0.010016</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>-0.020856</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.046371</td>\n",
       "      <td>0.017163</td>\n",
       "      <td>0.076185</td>\n",
       "      <td>0.057292</td>\n",
       "      <td>-0.002425</td>\n",
       "      <td>-0.011156</td>\n",
       "      <td>0.017117</td>\n",
       "      <td>-0.004184</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>BMI</th>\n",
       "      <td>0.109272</td>\n",
       "      <td>0.046371</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.012759</td>\n",
       "      <td>0.003750</td>\n",
       "      <td>0.198341</td>\n",
       "      <td>-0.138156</td>\n",
       "      <td>-0.135996</td>\n",
       "      <td>0.270025</td>\n",
       "      <td>-0.006205</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Children</th>\n",
       "      <td>0.042469</td>\n",
       "      <td>0.017163</td>\n",
       "      <td>0.012759</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.007673</td>\n",
       "      <td>0.067998</td>\n",
       "      <td>-0.022808</td>\n",
       "      <td>0.024806</td>\n",
       "      <td>-0.023066</td>\n",
       "      <td>0.021914</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Smoker</th>\n",
       "      <td>-0.025019</td>\n",
       "      <td>0.076185</td>\n",
       "      <td>0.003750</td>\n",
       "      <td>0.007673</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.787251</td>\n",
       "      <td>0.002811</td>\n",
       "      <td>-0.036945</td>\n",
       "      <td>0.068498</td>\n",
       "      <td>-0.036945</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Charges</th>\n",
       "      <td>0.299008</td>\n",
       "      <td>0.057292</td>\n",
       "      <td>0.198341</td>\n",
       "      <td>0.067998</td>\n",
       "      <td>0.787251</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.006349</td>\n",
       "      <td>-0.039905</td>\n",
       "      <td>0.073982</td>\n",
       "      <td>-0.043210</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>northeast</th>\n",
       "      <td>0.002475</td>\n",
       "      <td>-0.002425</td>\n",
       "      <td>-0.138156</td>\n",
       "      <td>-0.022808</td>\n",
       "      <td>0.002811</td>\n",
       "      <td>0.006349</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.320177</td>\n",
       "      <td>-0.345561</td>\n",
       "      <td>-0.320177</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>northwest</th>\n",
       "      <td>-0.000407</td>\n",
       "      <td>-0.011156</td>\n",
       "      <td>-0.135996</td>\n",
       "      <td>0.024806</td>\n",
       "      <td>-0.036945</td>\n",
       "      <td>-0.039905</td>\n",
       "      <td>-0.320177</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.346265</td>\n",
       "      <td>-0.320829</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>southeast</th>\n",
       "      <td>-0.011642</td>\n",
       "      <td>0.017117</td>\n",
       "      <td>0.270025</td>\n",
       "      <td>-0.023066</td>\n",
       "      <td>0.068498</td>\n",
       "      <td>0.073982</td>\n",
       "      <td>-0.345561</td>\n",
       "      <td>-0.346265</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.346265</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>southwest</th>\n",
       "      <td>0.010016</td>\n",
       "      <td>-0.004184</td>\n",
       "      <td>-0.006205</td>\n",
       "      <td>0.021914</td>\n",
       "      <td>-0.036945</td>\n",
       "      <td>-0.043210</td>\n",
       "      <td>-0.320177</td>\n",
       "      <td>-0.320829</td>\n",
       "      <td>-0.346265</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                Age       Sex       BMI  Children    Smoker   Charges  \\\n",
       "Age        1.000000 -0.020856  0.109272  0.042469 -0.025019  0.299008   \n",
       "Sex       -0.020856  1.000000  0.046371  0.017163  0.076185  0.057292   \n",
       "BMI        0.109272  0.046371  1.000000  0.012759  0.003750  0.198341   \n",
       "Children   0.042469  0.017163  0.012759  1.000000  0.007673  0.067998   \n",
       "Smoker    -0.025019  0.076185  0.003750  0.007673  1.000000  0.787251   \n",
       "Charges    0.299008  0.057292  0.198341  0.067998  0.787251  1.000000   \n",
       "northeast  0.002475 -0.002425 -0.138156 -0.022808  0.002811  0.006349   \n",
       "northwest -0.000407 -0.011156 -0.135996  0.024806 -0.036945 -0.039905   \n",
       "southeast -0.011642  0.017117  0.270025 -0.023066  0.068498  0.073982   \n",
       "southwest  0.010016 -0.004184 -0.006205  0.021914 -0.036945 -0.043210   \n",
       "\n",
       "           northeast  northwest  southeast  southwest  \n",
       "Age         0.002475  -0.000407  -0.011642   0.010016  \n",
       "Sex        -0.002425  -0.011156   0.017117  -0.004184  \n",
       "BMI        -0.138156  -0.135996   0.270025  -0.006205  \n",
       "Children   -0.022808   0.024806  -0.023066   0.021914  \n",
       "Smoker      0.002811  -0.036945   0.068498  -0.036945  \n",
       "Charges     0.006349  -0.039905   0.073982  -0.043210  \n",
       "northeast   1.000000  -0.320177  -0.345561  -0.320177  \n",
       "northwest  -0.320177   1.000000  -0.346265  -0.320829  \n",
       "southeast  -0.345561  -0.346265   1.000000  -0.346265  \n",
       "southwest  -0.320177  -0.320829  -0.346265   1.000000  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:48.141786Z",
     "iopub.status.busy": "2021-01-01T17:52:48.140923Z",
     "iopub.status.idle": "2021-01-01T17:52:48.149545Z",
     "shell.execute_reply": "2021-01-01T17:52:48.150035Z"
    },
    "papermill": {
     "duration": 0.080561,
     "end_time": "2021-01-01T17:52:48.150209",
     "exception": false,
     "start_time": "2021-01-01T17:52:48.069648",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ttest_indResult(statistic=46.664921172723716, pvalue=8.271435842177219e-283)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a=df[\"Charges\"].loc[df[\"Smoker\"]==1]\n",
    "b=df[\"Charges\"].loc[df[\"Smoker\"]==0]\n",
    "\n",
    "stats.ttest_ind(a,b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.063004,
     "end_time": "2021-01-01T17:52:48.277184",
     "exception": false,
     "start_time": "2021-01-01T17:52:48.214180",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* The charges can range from USD 1121 to USD 63770 with an average charge of USD 13270 and a right skewness of 1.51.\n",
    "* Charges are strongly correlated (0.79) and statistically significant (8e-283 pvalue) to whether the individual smokes or not.\n",
    "* Age and BMI are weakly correlated to charges; sex, number of children and region have almost no correlation with charges."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.065009,
     "end_time": "2021-01-01T17:52:48.407472",
     "exception": false,
     "start_time": "2021-01-01T17:52:48.342463",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The data will be split into a training and a testing set to train the model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:48.542335Z",
     "iopub.status.busy": "2021-01-01T17:52:48.541584Z",
     "iopub.status.idle": "2021-01-01T17:52:48.702967Z",
     "shell.execute_reply": "2021-01-01T17:52:48.701694Z"
    },
    "papermill": {
     "duration": 0.231369,
     "end_time": "2021-01-01T17:52:48.703115",
     "exception": false,
     "start_time": "2021-01-01T17:52:48.471746",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:48.840756Z",
     "iopub.status.busy": "2021-01-01T17:52:48.839691Z",
     "iopub.status.idle": "2021-01-01T17:52:48.843435Z",
     "shell.execute_reply": "2021-01-01T17:52:48.842889Z"
    },
    "papermill": {
     "duration": 0.075327,
     "end_time": "2021-01-01T17:52:48.843560",
     "exception": false,
     "start_time": "2021-01-01T17:52:48.768233",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x=df.drop([\"Charges\"],axis=1)\n",
    "y=df[\"Charges\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:48.980097Z",
     "iopub.status.busy": "2021-01-01T17:52:48.978874Z",
     "iopub.status.idle": "2021-01-01T17:52:48.984022Z",
     "shell.execute_reply": "2021-01-01T17:52:48.983301Z"
    },
    "papermill": {
     "duration": 0.077374,
     "end_time": "2021-01-01T17:52:48.984266",
     "exception": false,
     "start_time": "2021-01-01T17:52:48.906892",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:49.124722Z",
     "iopub.status.busy": "2021-01-01T17:52:49.123918Z",
     "iopub.status.idle": "2021-01-01T17:52:49.129180Z",
     "shell.execute_reply": "2021-01-01T17:52:49.129695Z"
    },
    "papermill": {
     "duration": 0.077172,
     "end_time": "2021-01-01T17:52:49.129876",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.052704",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set x shape: (896, 9) and y shape (896,)\n",
      "Testing set x shape: (442, 9) and y shape (442,)\n"
     ]
    }
   ],
   "source": [
    "print(\"Training set x shape:\",x_train.shape,\"and y shape\",y_train.shape)\n",
    "print(\"Testing set x shape:\",x_test.shape,\"and y shape\",y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.063941,
     "end_time": "2021-01-01T17:52:49.258931",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.194990",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "As the data includes a range of continuous and discrete variables, the data will need to be scaled to ensure those larger values to not give an inaccurate higher weighting. The training set will be fit to the scaler and transformed, while the testing set will only be transformed to avoid any data leakage:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:49.395159Z",
     "iopub.status.busy": "2021-01-01T17:52:49.394506Z",
     "iopub.status.idle": "2021-01-01T17:52:49.397596Z",
     "shell.execute_reply": "2021-01-01T17:52:49.398177Z"
    },
    "papermill": {
     "duration": 0.074404,
     "end_time": "2021-01-01T17:52:49.398337",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.323933",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:49.534244Z",
     "iopub.status.busy": "2021-01-01T17:52:49.533570Z",
     "iopub.status.idle": "2021-01-01T17:52:49.536932Z",
     "shell.execute_reply": "2021-01-01T17:52:49.536240Z"
    },
    "papermill": {
     "duration": 0.072172,
     "end_time": "2021-01-01T17:52:49.537050",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.464878",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "scaler=StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:49.677275Z",
     "iopub.status.busy": "2021-01-01T17:52:49.676557Z",
     "iopub.status.idle": "2021-01-01T17:52:49.683593Z",
     "shell.execute_reply": "2021-01-01T17:52:49.683003Z"
    },
    "papermill": {
     "duration": 0.080156,
     "end_time": "2021-01-01T17:52:49.683710",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.603554",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_train=scaler.fit_transform(x_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:49.823628Z",
     "iopub.status.busy": "2021-01-01T17:52:49.822930Z",
     "iopub.status.idle": "2021-01-01T17:52:49.827132Z",
     "shell.execute_reply": "2021-01-01T17:52:49.826451Z"
    },
    "papermill": {
     "duration": 0.075998,
     "end_time": "2021-01-01T17:52:49.827258",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.751260",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_test=scaler.transform(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.067028,
     "end_time": "2021-01-01T17:52:49.961082",
     "exception": false,
     "start_time": "2021-01-01T17:52:49.894054",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "There are nine independent variables that can predict the target variable, *Charges*, but not all nine will have the same predictive power. Based on their scores (larger is better) and pvalues (smaller is better) it can then be decided if all nine or a selection of the variables will be used in the model. If only a selection will be used - as with scaling - it is important to fit and transform the training set while only transforming the testing set to prevent any data leakage:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:50.101456Z",
     "iopub.status.busy": "2021-01-01T17:52:50.100328Z",
     "iopub.status.idle": "2021-01-01T17:52:50.308078Z",
     "shell.execute_reply": "2021-01-01T17:52:50.307322Z"
    },
    "papermill": {
     "duration": 0.278181,
     "end_time": "2021-01-01T17:52:50.308205",
     "exception": false,
     "start_time": "2021-01-01T17:52:50.030024",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import SelectKBest,f_regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:50.444574Z",
     "iopub.status.busy": "2021-01-01T17:52:50.443693Z",
     "iopub.status.idle": "2021-01-01T17:52:50.446816Z",
     "shell.execute_reply": "2021-01-01T17:52:50.446064Z"
    },
    "papermill": {
     "duration": 0.072605,
     "end_time": "2021-01-01T17:52:50.446943",
     "exception": false,
     "start_time": "2021-01-01T17:52:50.374338",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "selector=SelectKBest(f_regression,k=\"all\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:50.587558Z",
     "iopub.status.busy": "2021-01-01T17:52:50.586823Z",
     "iopub.status.idle": "2021-01-01T17:52:50.590927Z",
     "shell.execute_reply": "2021-01-01T17:52:50.590367Z"
    },
    "papermill": {
     "duration": 0.079314,
     "end_time": "2021-01-01T17:52:50.591048",
     "exception": false,
     "start_time": "2021-01-01T17:52:50.511734",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "x_selection=selector.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:50.728427Z",
     "iopub.status.busy": "2021-01-01T17:52:50.727507Z",
     "iopub.status.idle": "2021-01-01T17:52:50.732247Z",
     "shell.execute_reply": "2021-01-01T17:52:50.732895Z"
    },
    "papermill": {
     "duration": 0.07631,
     "end_time": "2021-01-01T17:52:50.733067",
     "exception": false,
     "start_time": "2021-01-01T17:52:50.656757",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'northeast', 'northwest',\n",
       "       'southeast', 'southwest'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:50.874555Z",
     "iopub.status.busy": "2021-01-01T17:52:50.873735Z",
     "iopub.status.idle": "2021-01-01T17:52:50.877746Z",
     "shell.execute_reply": "2021-01-01T17:52:50.877135Z"
    },
    "papermill": {
     "duration": 0.078336,
     "end_time": "2021-01-01T17:52:50.877867",
     "exception": false,
     "start_time": "2021-01-01T17:52:50.799531",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7.94849923e+01, 3.46815689e+00, 4.36805508e+01, 2.83600501e+00,\n",
       "       1.44342260e+03, 6.82315883e-02, 7.39688840e-01, 5.68765383e+00,\n",
       "       3.53076642e+00])"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_selection.scores_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:51.019343Z",
     "iopub.status.busy": "2021-01-01T17:52:51.018264Z",
     "iopub.status.idle": "2021-01-01T17:52:51.022049Z",
     "shell.execute_reply": "2021-01-01T17:52:51.022635Z"
    },
    "papermill": {
     "duration": 0.077964,
     "end_time": "2021-01-01T17:52:51.022796",
     "exception": false,
     "start_time": "2021-01-01T17:52:50.944832",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2.68888736e-018, 6.28888600e-002, 6.63666856e-011, 9.25221932e-002,\n",
       "       8.97113856e-189, 7.93989365e-001, 3.89991145e-001, 1.72922260e-002,\n",
       "       6.05650683e-002])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_selection.pvalues_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.069396,
     "end_time": "2021-01-01T17:52:51.158701",
     "exception": false,
     "start_time": "2021-01-01T17:52:51.089305",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "* As discovered previously, smoking is a strong predictor with the largest score (1e+03) and smallest pvalue (8e-189). This is followed by age and BMI.\n",
    "* All of the nine variables are statistically significant with a pvalue of less than 0.05, so all the nine variables will be used in the model. Hence no need to transform the data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.068161,
     "end_time": "2021-01-01T17:52:51.295921",
     "exception": false,
     "start_time": "2021-01-01T17:52:51.227760",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Various regression models will be tested and evaluated based on their cross validated accuracy score:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:51.436545Z",
     "iopub.status.busy": "2021-01-01T17:52:51.435787Z",
     "iopub.status.idle": "2021-01-01T17:52:51.508692Z",
     "shell.execute_reply": "2021-01-01T17:52:51.507870Z"
    },
    "papermill": {
     "duration": 0.14532,
     "end_time": "2021-01-01T17:52:51.508825",
     "exception": false,
     "start_time": "2021-01-01T17:52:51.363505",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.svm import SVR\n",
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:51.652320Z",
     "iopub.status.busy": "2021-01-01T17:52:51.651652Z",
     "iopub.status.idle": "2021-01-01T17:52:51.655178Z",
     "shell.execute_reply": "2021-01-01T17:52:51.654626Z"
    },
    "papermill": {
     "duration": 0.07808,
     "end_time": "2021-01-01T17:52:51.655617",
     "exception": false,
     "start_time": "2021-01-01T17:52:51.577537",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:51.797231Z",
     "iopub.status.busy": "2021-01-01T17:52:51.796508Z",
     "iopub.status.idle": "2021-01-01T17:52:51.800331Z",
     "shell.execute_reply": "2021-01-01T17:52:51.799646Z"
    },
    "papermill": {
     "duration": 0.077688,
     "end_time": "2021-01-01T17:52:51.800465",
     "exception": false,
     "start_time": "2021-01-01T17:52:51.722777",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "names=[\"LinearRegression\",\"kNN\",\"SVM\",\"RandomForest\"]\n",
    "models=[LinearRegression(),KNeighborsRegressor(),SVR(),RandomForestRegressor()]\n",
    "meancross=[]\n",
    "stdcross=[]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:51.953112Z",
     "iopub.status.busy": "2021-01-01T17:52:51.951964Z",
     "iopub.status.idle": "2021-01-01T17:52:54.561028Z",
     "shell.execute_reply": "2021-01-01T17:52:54.561543Z"
    },
    "papermill": {
     "duration": 2.694637,
     "end_time": "2021-01-01T17:52:54.561725",
     "exception": false,
     "start_time": "2021-01-01T17:52:51.867088",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "for model in models:\n",
    "    model=model\n",
    "    model.fit(x_train,y_train)\n",
    "    rcross=cross_val_score(model,x_train,y_train,cv=5)\n",
    "    meancross.append(rcross.mean())\n",
    "    stdcross.append(rcross.std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:54.709280Z",
     "iopub.status.busy": "2021-01-01T17:52:54.708159Z",
     "iopub.status.idle": "2021-01-01T17:52:54.773247Z",
     "shell.execute_reply": "2021-01-01T17:52:54.772561Z"
    },
    "papermill": {
     "duration": 0.138735,
     "end_time": "2021-01-01T17:52:54.773398",
     "exception": false,
     "start_time": "2021-01-01T17:52:54.634663",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Name</th>\n",
       "      <th>Model</th>\n",
       "      <th>RCrossMean</th>\n",
       "      <th>RCrossStd</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>RandomForest</td>\n",
       "      <td>(DecisionTreeRegressor(max_features='auto', ra...</td>\n",
       "      <td>0.843695</td>\n",
       "      <td>0.027661</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>kNN</td>\n",
       "      <td>KNeighborsRegressor()</td>\n",
       "      <td>0.765199</td>\n",
       "      <td>0.018811</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LinearRegression</td>\n",
       "      <td>LinearRegression()</td>\n",
       "      <td>0.738096</td>\n",
       "      <td>0.028232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SVM</td>\n",
       "      <td>SVR()</td>\n",
       "      <td>-0.098294</td>\n",
       "      <td>0.026054</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Name                                              Model  \\\n",
       "3      RandomForest  (DecisionTreeRegressor(max_features='auto', ra...   \n",
       "1               kNN                              KNeighborsRegressor()   \n",
       "0  LinearRegression                                 LinearRegression()   \n",
       "2               SVM                                              SVR()   \n",
       "\n",
       "   RCrossMean  RCrossStd  \n",
       "3    0.843695   0.027661  \n",
       "1    0.765199   0.018811  \n",
       "0    0.738096   0.028232  \n",
       "2   -0.098294   0.026054  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results=pd.DataFrame({\"Name\":names,\"Model\":models,\"RCrossMean\":meancross,\"RCrossStd\":stdcross})\n",
    "results.sort_values(\"RCrossMean\",ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.070304,
     "end_time": "2021-01-01T17:52:54.911887",
     "exception": false,
     "start_time": "2021-01-01T17:52:54.841583",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Although the RandomForest model has the second largest spread of cross validation scores, the RandomForest performed the best so let's explore more ensemble models:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:55.065532Z",
     "iopub.status.busy": "2021-01-01T17:52:55.064450Z",
     "iopub.status.idle": "2021-01-01T17:52:55.067850Z",
     "shell.execute_reply": "2021-01-01T17:52:55.067221Z"
    },
    "papermill": {
     "duration": 0.084273,
     "end_time": "2021-01-01T17:52:55.067982",
     "exception": false,
     "start_time": "2021-01-01T17:52:54.983709",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:55.214129Z",
     "iopub.status.busy": "2021-01-01T17:52:55.212993Z",
     "iopub.status.idle": "2021-01-01T17:52:55.216504Z",
     "shell.execute_reply": "2021-01-01T17:52:55.215808Z"
    },
    "papermill": {
     "duration": 0.079374,
     "end_time": "2021-01-01T17:52:55.216625",
     "exception": false,
     "start_time": "2021-01-01T17:52:55.137251",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "names=[\"RandomForest\",\"Bagging\",\"GradientBoost\",\"AdaBoost\"]\n",
    "models=[RandomForestRegressor(),BaggingRegressor(),GradientBoostingRegressor(),AdaBoostRegressor()]\n",
    "meancross=[]\n",
    "stdcross=[]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:55.410748Z",
     "iopub.status.busy": "2021-01-01T17:52:55.379148Z",
     "iopub.status.idle": "2021-01-01T17:52:58.624710Z",
     "shell.execute_reply": "2021-01-01T17:52:58.623932Z"
    },
    "papermill": {
     "duration": 3.337649,
     "end_time": "2021-01-01T17:52:58.624858",
     "exception": false,
     "start_time": "2021-01-01T17:52:55.287209",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "for model in models:\n",
    "    model=model\n",
    "    model.fit(x_train,y_train)\n",
    "    rcross=cross_val_score(model,x_train,y_train,cv=5)\n",
    "    meancross.append(rcross.mean())\n",
    "    stdcross.append(rcross.std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:58.831416Z",
     "iopub.status.busy": "2021-01-01T17:52:58.810579Z",
     "iopub.status.idle": "2021-01-01T17:52:58.989387Z",
     "shell.execute_reply": "2021-01-01T17:52:58.988237Z"
    },
    "papermill": {
     "duration": 0.288536,
     "end_time": "2021-01-01T17:52:58.989528",
     "exception": false,
     "start_time": "2021-01-01T17:52:58.700992",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Name</th>\n",
       "      <th>Model</th>\n",
       "      <th>RCrossMean</th>\n",
       "      <th>RCrossStd</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>GradientBoost</td>\n",
       "      <td>([DecisionTreeRegressor(criterion='friedman_ms...</td>\n",
       "      <td>0.859443</td>\n",
       "      <td>0.026130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>RandomForest</td>\n",
       "      <td>(DecisionTreeRegressor(max_features='auto', ra...</td>\n",
       "      <td>0.841420</td>\n",
       "      <td>0.027975</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Bagging</td>\n",
       "      <td>(DecisionTreeRegressor(random_state=1686213370...</td>\n",
       "      <td>0.826649</td>\n",
       "      <td>0.023138</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AdaBoost</td>\n",
       "      <td>(DecisionTreeRegressor(max_depth=3, random_sta...</td>\n",
       "      <td>0.812949</td>\n",
       "      <td>0.028299</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Name                                              Model  \\\n",
       "2  GradientBoost  ([DecisionTreeRegressor(criterion='friedman_ms...   \n",
       "0   RandomForest  (DecisionTreeRegressor(max_features='auto', ra...   \n",
       "1        Bagging  (DecisionTreeRegressor(random_state=1686213370...   \n",
       "3       AdaBoost  (DecisionTreeRegressor(max_depth=3, random_sta...   \n",
       "\n",
       "   RCrossMean  RCrossStd  \n",
       "2    0.859443   0.026130  \n",
       "0    0.841420   0.027975  \n",
       "1    0.826649   0.023138  \n",
       "3    0.812949   0.028299  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results=pd.DataFrame({\"Name\":names,\"Model\":models,\"RCrossMean\":meancross,\"RCrossStd\":stdcross})\n",
    "results.sort_values(\"RCrossMean\",ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.069099,
     "end_time": "2021-01-01T17:52:59.129169",
     "exception": false,
     "start_time": "2021-01-01T17:52:59.060070",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Out of the ensemble models, GradientBoost performed the best even though - again - it has the second largest spread of cross validation scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:59.282312Z",
     "iopub.status.busy": "2021-01-01T17:52:59.280803Z",
     "iopub.status.idle": "2021-01-01T17:52:59.396203Z",
     "shell.execute_reply": "2021-01-01T17:52:59.395416Z"
    },
    "papermill": {
     "duration": 0.196058,
     "end_time": "2021-01-01T17:52:59.396335",
     "exception": false,
     "start_time": "2021-01-01T17:52:59.200277",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model=GradientBoostingRegressor()\n",
    "model.fit(x_train,y_train)\n",
    "y_predict=model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:52:59.562631Z",
     "iopub.status.busy": "2021-01-01T17:52:59.555125Z",
     "iopub.status.idle": "2021-01-01T17:52:59.864959Z",
     "shell.execute_reply": "2021-01-01T17:52:59.864092Z"
    },
    "papermill": {
     "duration": 0.391667,
     "end_time": "2021-01-01T17:52:59.865114",
     "exception": false,
     "start_time": "2021-01-01T17:52:59.473447",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 60000.0)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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LVvYuHFj37UGS3f8wgsvsH///r6fOzs76erqYvPmzfz4x8ktFu+66y6Ki4uvKFteXs6cOXPYsmUL8+bN49ChQ2zYsAGAr33ta33J1s6cOcORI0fSCvTd3d28+uqr3H///X3HotEoABs2bOChhx7id3/3d/uybI6nYQO9iBQD9wC1QDvwIxH5oDHmB/3PM8Y8CjwKyVw3GairUmocXByjv9zFxGCQHKL5+te/zh133HHJOc8+++ywqYzTTXfseR6vvfYaoVDoivfSKf/AAw/wn//5nyxcuJB7770XEeHll1/mhRde4LXXXiMnJ4dbbrnlijTH/VMcw1uplj3Po6ioaMDP5pFHHmHr1q0888wzrFixgl27dqX9V0ImpPMw9u3ACWNMszEmDvwYGNljcKVUVrvjjjv4xje+0Zf+9/Dhw/T09LBx40Yef/xxXNeloaGBl1566Yqy69ev59e//nVf1snB0h1v2rTpko2+LwbYjRs38sMf/hCAX/ziF1y4cGHAOt533308/fTTPPbYY32blnR0dFBcXExOTg4HDx5ky5YtV5SrrKykqamJ1tZWotEoP//5zwEoKCigtraWH/3oR0DyF9bu3buB5Nj99ddfz5e+9CXKyso4c+ZMuh9lRqQzRn8auEFEckgO3dwOZD41pVLqChM1A+uHP/xhTp48yapVqzDGUF5eztNPP829997Liy++SF1dHfPnz+fmm2++omx5eTmPPvoo9913H57nUVFRwa9+9Sve9a538b73vY+f/OQnffvRfuxjH2PZsmUkEgk2btzII488wuc//3ne//73s2rVKm6++eZBh46Li4tZvHgx+/fvZ926dQDceeedPPLIIyxbtowFCxZwww03XFHOcRw+97nPcf3111NbW8vChQv73vvhD3/In/zJn/DXf/3XxONxHnzwQZYvX86nP/1pjhw5gjGG22+/neXLl4/SJ3110kpTLCJfBB4AEsBO4MPGmOhg52uaYqVGh6Ypzk5jnaY4rVk3xpjPA5/PRAWUUkpllqZAUEqpLKeBXqkJLhO7wKnxMx7/PTXQKzWBBYNBWltbNdhnCWMMra2tBINjm0VGk5opNYFVV1dTX19Pc3PzeFdFjZJgMEh1dfWY3lMDvVITmOM4fStGlbpaOnSjlFJZTgO9UkplOQ30SimV5TTQK6VUltNAr5RSWU4DvVJKZbkx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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(y_test,color=\"yellowgreen\",label=\"Actual values\")\n",
    "sns.distplot(y_predict,color=\"lightcoral\",label=\"Predicted values\")\n",
    "plt.legend()\n",
    "plt.xlim([0,60000])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:53:00.031320Z",
     "iopub.status.busy": "2021-01-01T17:53:00.023899Z",
     "iopub.status.idle": "2021-01-01T17:53:00.371704Z",
     "shell.execute_reply": "2021-01-01T17:53:00.370932Z"
    },
    "papermill": {
     "duration": 0.429708,
     "end_time": "2021-01-01T17:53:00.371831",
     "exception": false,
     "start_time": "2021-01-01T17:52:59.942123",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Predicted Price')"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "f,axes=plt.subplots(2,1,sharex=True)\n",
    "sns.boxplot(y_test,color=\"yellowgreen\",whis=5,ax=axes[0])\n",
    "sns.boxplot(y_predict,color=\"lightcoral\",whis=5,ax=axes[1])\n",
    "axes[0].set_xlabel(\"\")\n",
    "plt.xlabel(\"Charges\")\n",
    "axes[0].set_ylabel(\"Actual Price\")\n",
    "axes[1].set_ylabel(\"Predicted Price\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:53:00.523163Z",
     "iopub.status.busy": "2021-01-01T17:53:00.522193Z",
     "iopub.status.idle": "2021-01-01T17:53:00.526926Z",
     "shell.execute_reply": "2021-01-01T17:53:00.526160Z"
    },
    "papermill": {
     "duration": 0.084259,
     "end_time": "2021-01-01T17:53:00.527055",
     "exception": false,
     "start_time": "2021-01-01T17:53:00.442796",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The model's score is 0.86.\n"
     ]
    }
   ],
   "source": [
    "print(\"The model's score is {:.2f}.\".format(model.score(x_test,y_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:53:00.691399Z",
     "iopub.status.busy": "2021-01-01T17:53:00.690593Z",
     "iopub.status.idle": "2021-01-01T17:53:00.695342Z",
     "shell.execute_reply": "2021-01-01T17:53:00.694682Z"
    },
    "papermill": {
     "duration": 0.084899,
     "end_time": "2021-01-01T17:53:00.695486",
     "exception": false,
     "start_time": "2021-01-01T17:53:00.610587",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ttest_indResult(statistic=-0.12839025533826162, pvalue=0.8978694110724472)"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a=y_test\n",
    "b=y_predict\n",
    "\n",
    "stats.ttest_ind(a,b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:53:00.852867Z",
     "iopub.status.busy": "2021-01-01T17:53:00.852088Z",
     "iopub.status.idle": "2021-01-01T17:53:00.872403Z",
     "shell.execute_reply": "2021-01-01T17:53:00.871684Z"
    },
    "papermill": {
     "duration": 0.103607,
     "end_time": "2021-01-01T17:53:00.872535",
     "exception": false,
     "start_time": "2021-01-01T17:53:00.768928",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Min</th>\n",
       "      <th>Max</th>\n",
       "      <th>Mean</th>\n",
       "      <th>Standard Deviation</th>\n",
       "      <th>Skew</th>\n",
       "      <th>Kurtosis</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Actual</th>\n",
       "      <td>1137.469700</td>\n",
       "      <td>51194.559140</td>\n",
       "      <td>13177.182438</td>\n",
       "      <td>11975.982235</td>\n",
       "      <td>1.453435</td>\n",
       "      <td>1.221367</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Predicted</th>\n",
       "      <td>458.333761</td>\n",
       "      <td>50188.098947</td>\n",
       "      <td>13277.399534</td>\n",
       "      <td>11219.590463</td>\n",
       "      <td>1.739549</td>\n",
       "      <td>2.246053</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   Min           Max          Mean  Standard Deviation  \\\n",
       "Actual     1137.469700  51194.559140  13177.182438        11975.982235   \n",
       "Predicted   458.333761  50188.098947  13277.399534        11219.590463   \n",
       "\n",
       "               Skew  Kurtosis  \n",
       "Actual     1.453435  1.221367  \n",
       "Predicted  1.739549  2.246053  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results=pd.DataFrame({\"Actual\":y_test,\"Predicted\":y_predict})\n",
    "\n",
    "smin=[stats.tmin(results[i]) for i in results.columns]\n",
    "smax=[stats.tmax(results[i]) for i in results.columns]\n",
    "smean=[stats.tmean(results[i]) for i in results.columns]\n",
    "sstd=[stats.tstd(results[i]) for i in results.columns]\n",
    "sskew=[stats.skew(results[i]) for i in results.columns]\n",
    "skurtosis=[stats.kurtosis(results[i]) for i in results.columns]\n",
    "\n",
    "stable=pd.DataFrame({\"Min\":smin,\"Max\":smax,\"Mean\":smean,\"Standard Deviation\":sstd,\"Skew\":sskew,\"Kurtosis\":skurtosis},index=results.columns)\n",
    "stable"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.077606,
     "end_time": "2021-01-01T17:53:01.026313",
     "exception": false,
     "start_time": "2021-01-01T17:53:00.948707",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "The base model gave an accuracy score of almost 86% and can be improved by obtaining more data, conducting further feature selection (as we utilised all the features) and perfoming parameter tuning (we have only used the default parameters). Nevertheless the actual and predicted resulare not statistically different as while the minimum value is rather off, the maximum value, mean, standard deviation, skew and kurtosis of the actual and predicted values are still similar."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.074056,
     "end_time": "2021-01-01T17:53:01.173337",
     "exception": false,
     "start_time": "2021-01-01T17:53:01.099281",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Try to predict your medical costs by inputting the relevant values in place for the '0's below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:53:01.332058Z",
     "iopub.status.busy": "2021-01-01T17:53:01.331339Z",
     "iopub.status.idle": "2021-01-01T17:53:01.335094Z",
     "shell.execute_reply": "2021-01-01T17:53:01.334577Z"
    },
    "papermill": {
     "duration": 0.086073,
     "end_time": "2021-01-01T17:53:01.335222",
     "exception": false,
     "start_time": "2021-01-01T17:53:01.249149",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "age=[0] # your age as a whole number\n",
    "sex=[0] # 0 if you are a female OR 1 if you are a male\n",
    "bmi=[0] # your calculated BMI as a whole or decimal number\n",
    "children=[0] # your number of children/dependents as a whole number\n",
    "smoker=[0] # 0 if you do not smoke OR 1 if you do smoke\n",
    "northeast=[0] # 0 if you live in the northwest/southeast/southwest OR 1 if you live in the northeast\n",
    "northwest=[0] # 0 if you live in the northeast/southeast/southwest OR 1 if you live in the northwest\n",
    "southeast=[0] # 0 if you live in the northeast/northwest/southwest OR 1 if you live in the southeast\n",
    "southwest=[0] # 0 if you live in the northeast/northwest/southeast OR 1 if you live in the southwest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-01-01T17:53:01.498042Z",
     "iopub.status.busy": "2021-01-01T17:53:01.497070Z",
     "iopub.status.idle": "2021-01-01T17:53:01.501818Z",
     "shell.execute_reply": "2021-01-01T17:53:01.501068Z"
    },
    "papermill": {
     "duration": 0.093445,
     "end_time": "2021-01-01T17:53:01.501947",
     "exception": false,
     "start_time": "2021-01-01T17:53:01.408502",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Your predicted medical costs is USD x.\n"
     ]
    }
   ],
   "source": [
    "if (age==[0])&(sex==[0])&(bmi==[0])&(children==[0])&(smoker==[0])&(northeast==[0])&(northwest==[0])&(southeast==[0])&(southwest==[0]):\n",
    "    print(\"Your predicted medical costs is USD x.\")\n",
    "else:\n",
    "    your_data=pd.DataFrame({\"Age\":age,\"Sex\":sex,\"BMI\":bmi,\"Children\":children,\"Smoker\":smoker,\"northeast\":northeast,\"northwest\":northwest,\"southeast\":southeast,\"southwest\":southwest})\n",
    "    your_data=scaler.transform(your_data)\n",
    "    your_costs=model.predict(your_data)\n",
    "    print(\"Your predicted medical costs is USD {:.2f}.\".format(your_costs[0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "papermill": {
     "duration": 0.080785,
     "end_time": "2021-01-01T17:53:01.662013",
     "exception": false,
     "start_time": "2021-01-01T17:53:01.581228",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Does this value seem right to you?"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  },
  "papermill": {
   "duration": 27.556368,
   "end_time": "2021-01-01T17:53:01.848366",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2021-01-01T17:52:34.291998",
   "version": "2.1.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
