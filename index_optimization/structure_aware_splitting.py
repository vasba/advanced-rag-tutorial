text = """
Charles Dickens was born in Portsmouth, England, on February 7, 1812. He was the second of eight children and his family moved to London when he was a young boy. Dickens' father was imprisoned for debt in 1824, and young Charles was forced to work in a blacking factory, a harsh and unpleasant experience that would later influence his writing.
After his father's release from prison, Dickens received some education but was largely self-taught. He began working as a journalist in his early twenties and soon gained recognition for his vivid descriptions of London life. His first novel, The Pickwick Papers, was published in installments in 1836-37 and was an immediate success.
Dickens went on to write many other popular novels, including Oliver Twist (1838), David Copperfield (1850), Bleak House (1853), A Tale of Two Cities (1859), and Great Expectations (1861). His novels are known for their social commentary, humor, and memorable characters. He is considered one of the greatest novelists of the English language.
Dickens died in Gad's Hill Place, England, on June 9, 1870.
"""

docs = text.split(".")

for doc in docs:
  print(doc.strip())