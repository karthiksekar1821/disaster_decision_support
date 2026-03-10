from preprocessing import clean_text, is_low_information

sample_tweets = [
    "Bridge collapsed near river! People trapped!! #disaster http://news.com",
    "@user Flood everywhere!!! Need help urgently!!!",
    "Help",
    "Pray for the victims 🙏 #support"
]

for tweet in sample_tweets:
    cleaned = clean_text(tweet)
    low_info = is_low_information(cleaned)
    
    print("Original:", tweet)
    print("Cleaned :", cleaned)
    print("Low Info:", low_info)
    print("-" * 50)