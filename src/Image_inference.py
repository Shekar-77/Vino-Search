from Images.Image import Image_vector_store
from PIL import Image
import matplotlib.pyplot as plt
# 1. Run the engine
response_engine = Image_vector_store(
    collection_name="new",
    folder_path="Sample_images",
    device='cpu'
)

# 2. Get the search points
response_engine.creating_vector_store()
result = response_engine.image_retrieval(query="What are images about?")
print(f"The result is:{result}")
# 3. Display the top match
if result:

    print("Got in ")
    top_hit = result[0] # The most similar image
    img_path = top_hit.payload['image_path']
    
    print(f"Found it! Showing: {img_path}")
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"Score: {top_hit.score:.2f}")
    plt.axis('off')
    plt.show()
