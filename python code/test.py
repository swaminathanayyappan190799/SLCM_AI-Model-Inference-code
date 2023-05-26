from PyQt5.QtWidgets import QApplication, QMessageBox

# Create the application
app = QApplication([])

# Create a message box dialog
message_box = QMessageBox()
message_box.setIcon(QMessageBox.Information)
message_box.setText("This is an informational message.")
message_box.setWindowTitle("Information")
message_box.setStandardButtons(QMessageBox.Ok)

# Show the message box
message_box.exec_()

# Run the application event loop
app.exec_()
