import random

COUNTRIES = [
    "Madagascar", "Taiwan", "USA", "Germany", "France", "Spain", "Russia", "China", 
    "Japan", "Brazil", "India", "Egypt", "South Africa", "Australia", "Canada", 
    "Mexico", "Indonesia", "Nigeria", "Turkey", "United Kingdom", "Italy", "Poland", 
    "Argentina", "Netherlands", "Belgium", "Switzerland", "Sweden", "Norway", "Finland",
    "Denmark", "Portugal", "Greece", "Iran", "Thailand", "Philippines", "Vietnam", 
    "South Korea", "Saudi Arabia", "Israel", "UAE", "New Zealand", "Ireland", "Malaysia",
    "Singapore", "Hong Kong", "Czech Republic", "Hungary", "Romania", "Colombia", 
    "Peru", "Venezuela", "Chile", "Morocco", "Algeria", "Tunisia", "Nepal", "Pakistan", "Bangladesh", 
    "Kazakhstan", "Ukraine", "Austria", "Croatia", "Serbia", "Kenya", "Ghana", "Zimbabwe",
    "Cuba", "Panama", "Fiji", "Mongolia", "North Korea", "Myanmar", "Ethiopia", "Tanzania",
    "Algeria", "Libya", "Jordan", "Qatar", "Oman", "Kuwait", "Lebanon", "Bulgaria", "Slovakia",
    "Lithuania", "Latvia", "Estonia", "Cyprus", "Luxembourg", "Macao", "Bhutan", "Maldives",
    "Angola", "Cameroon", "Senegal", "Mali", "Zambia", "Uganda", "Namibia", "Botswana",
    "Mozambique", "Ivory Coast", "Burkina Faso", "Malawi", "Gabon", "Lesotho", "Gambia",
    "Guinea", "Cape Verde", "Rwanda", "Benin", "Burundi", "Somalia", "Eritrea", "Djibouti",
    "Togo", "Seychelles", "Chad", "Central African Republic", "Liberia", "Mauritania", "Sri Lanka",
    "Sierra Leone", "Equatorial Guinea", "Swaziland", "Congo (Kinshasa)", "Congo (Brazzaville)"
]

TEXT_TYPES = [
    "Passive Voice", "Casual sentence", "In the past", "Do you know", "A character appears in the middle or end of the sentence, emphasizing the relative clause"
    "Have you meet", "Inversion for Emphasis", "I have heard about", "Sentence begins with a adverbial phrase, the character appears in the middle of the sentence"

]

FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "Matteo", "Richard", "Joseph", "Charles", "Lorenzo", 
    "Christopher", "Aurora", "Matthew", "Anthony", "Donald", "Paul", "Mark", "Steven", "Andrew", "Kenneth", 
    "George", "Brian", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Frank", "Kevin", "Ivan", 
    "Alexander", "Dmitry", "Sergey", "Mikhail", "Vladimir", "Andrei", "Nikolai", "Igor", "Pavel", "Sophia", 
    "Marcos", "Maria", "Hannah", "Ana", "Lucas", "Leon", "Pedro", "Noah", "Emma", "Xu", "Sun", "Ma", "Zhu", 
    "Lin", "Guo", "Cheng", "He", "Luo", "Cao", "María", "José", "Juan", "Carlos", "Francisco", "Antonio", 
    "Manuel", "Alejandro", "Diego", "Miguel", "David", "Javier", "Felix", "Rafael", "Fernando", "Luis", 
    "Daniel", "Beatrice", "Eduardo", "Alberto", "Roberto", "Pablo", "Sergio", "Jorge", "Oscar", "Ricardo", 
    "Ángel", "Santiago", "Emilio", "Domingo", "Andrés", "Muhammad", "Ahmed", "Ali", "Omar", "Hassan", 
    "Ibrahim", "Max", "Karim", "Hussein", "Mahmoud" 
]

SECOND_NAMES = [
    "Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito", "Yamamoto", "Nakamura", "Kobayashi", 
    "Kato", "Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Yoo", "Jang", "Ahn", "Song", "Smith", 
    "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Martinez", 
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", 
    "White", "Papadopoulos", "Giannakopoulos", "Georgiou", "Dimitriou", "Nikolaidis", "Angelopoulos", 
    "Petrou", "Ioannidis", "Christodoulou", "Konstantinou", "Vlachos", "Karagiannis", "Alexandrou", 
    "Kouros", "Manolakis", "Tsiamis", "Zacharias", "Tsakiris", "Panagiotopoulos", "Stefanidis", 
    "Ivanov", "Smirnov", "Kuznetsov", "Popov", "Sokolov", "Vasiliev", "Petrov", "Sidorov", "Mikhailov", 
    "Nikolaev", "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou", "Kenyatta", 
    "Otieno", "Mwangi", "Njoroge", "Kiplagat", "Ndlovu", "Zulu", "Dlamini", "Mthembu", "Sithole", "Diop", 
    "Faye", "Mohamed", "Farouk", "Amin", "Youssef", "Salah", "Gaber", "Shawky", "Tawfik", "El Sayed" 
]

def find_duplicates(arr):
    return set([x for x in arr if arr.count(x) > 1])

def find_common_elements(arr1, arr2):
    # Convert arrays to sets for faster lookup
    set1 = set(arr1)
    set2 = set(arr2)
    # Find intersection of the sets (common elements)
    common_elements = set1.intersection(set2)
    return list(common_elements)


if __name__ == "__main__":
    print("--data statistics--")

    names_duplicates = find_duplicates(FIRST_NAMES)
    print(f"There are {len(names_duplicates)} duplicates in names: {names_duplicates}")

    surnames_duplicates = find_duplicates(SECOND_NAMES)
    print(f"There are {len(surnames_duplicates)} duplicates in surnames: {surnames_duplicates}")

    common_elements = find_common_elements(FIRST_NAMES, SECOND_NAMES)
    print(f"There are {len(common_elements)} common elements: {common_elements}")

    print(f"There are {len(FIRST_NAMES)} names, and {len(SECOND_NAMES)} surnames")
    assert len(FIRST_NAMES) == len(SECOND_NAMES)

    print("--data examples--")
    random_samples = 5
    for i in range(random_samples):
        name = random.choice(FIRST_NAMES)
        surname = random.choice(SECOND_NAMES)

        print(f"{i+1}: {name} {surname}")
