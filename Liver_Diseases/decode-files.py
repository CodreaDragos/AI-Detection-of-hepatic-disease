import pydicom
import matplotlib.pyplot as plt

# Load the DICOM file
dicom_file = pydicom.dcmread(r"C:\Users\Cobra\Desktop\fac\an3sem2\ira2\proiect ira2\Liver_Diseases\studies\8\1.2.392.200036.9116.4.2.8859.1065.20221031124451893.1.2316")

# Display the DICOM image
plt.imshow(dicom_file.pixel_array, cmap='gray')
plt.show()