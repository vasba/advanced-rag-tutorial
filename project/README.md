# Running the Project

WARNING: This assumes very good knowledge of Python

## From Visual Studio Code

1. **Open the project in Visual Studio Code**:
   - Open the folder containing your project files in Visual Studio Code.

2. **Configure the debugger**:
   - Ensure that the `.vscode/launch.json` file is configured correctly. It should look like this:
     ```json
     {
         "version": "0.1.0",
         "configurations": [
             {
                 "name": "debug streamlit",
                 "type": "debugpy",
                 "request": "launch",
                 "program": "your path to streamlit",
                 "args": [
                     "run",
                     "project/app.py"
                 ]
             }
         ]
     }
     ```

3. **Start debugging**:
   - Press `F5` or go to the Run and Debug view (`Ctrl+Shift+D`) and select the "debug streamlit" configuration. This will start the Streamlit application.

## From the Command Line

1. **Navigate to the project directory**:
   ```sh
   cd path/to/project_folder_in_this_repo
   ```

2. **Set up a virtual environment (if not already set up)**:
    ```sh
    python -m venv venv
    ```
3. **Activate the virtual environment**:
    - On Windows:
    ```sh
    .\venv\Scripts\activate
    ```
    - On macOS/Linux:
    ```sh
    source venv/bin/activate
    ```
4. **Install the required dependencies**:
    - Ensure you have a virtual environment activated. Then, install the dependencies listed in requirements.txt:

    ```sh
    pip install -r requirements.txt
    ```

5. **Run the Streamlit application**:
    - Execute the following command to start the Streamlit app:

    ```sh
    streamlit run app.py
    ```

This will start the Streamlit application, and you can access it in your web browser at the URL provided in the terminal output.

