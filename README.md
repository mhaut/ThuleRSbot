# Analysis of remote sensing images through social tools

![](https://img.shields.io/github/stars/mhaut/ThuleRSbot.svg) ![](https://img.shields.io/github/forks/mhaut/ThuleRSbot.svg) ![](https://img.shields.io/github/issues/mhaut/ThuleRSbot.svg) 

The Code for "Analysis of remote sensing images through social tools".

```
A. Redondo, J. M. Haut, M. E. Paoletti, J. Plaza and A. Plaza.
Analysis of remote sensing images through social tools
```

### Install requires packages (with conda)
```
cd environmentConda
conda env create -f enviroment.yml (without cuda)
conda env create -f enviromentCUDA.yml (with cuda support)
```

### Bot configuration (etc/config.json)

1. Bot configuration (Required). <br/>
   **parallelOrQueueExecutionAlg**: this variable is the bot's execution mode and will have "parallel" (if you want the executions to be parallel between the users) or "queue" (if you want the executions to be one by one in order of arrival and priority).

2. Telegram configuration (Required).<br/>
    To configure Telegram it is necessary to create a bot in Telegram and then obtain the token that is associated with the created bot and the account of the creator of the bot. To manage the bot (create it, get the token, put the name, etc.) [@BotFather](https://t.me/botfather) is used. For more information visit [Telegram Bot](https://core.telegram.org/bots).<br/>
        **API_TOKEN**: contains the obtained token. It is only necessary to delete the content of the variable (Insert Token) and paste the token between the quotes (the token is a string).<br/>
        **num_Threads**: contains the number of threads that the bot will have to manage all users (the minimum value is 2).

3. SentinelHub Configuration (Optional). <br/>
    If you want to download the satellite images from this platform, you need to create an account and configure it to obtain the satellite images. For more information visit [SentienelHub](https://www.sentinel-hub.com/develop/dashboard/).<br/>
    **API_TOKEN**: will contain the token associated with the SentinelHub configuration.

4. Email (Optional). <br/>
    The bot has the ability to send the results by mail. For this it is necessary to fill in the following fields (the parameters are all string and must be enclosed in quotation marks):<br/>
    **from_addrs**: contains the email address with which the results will be sent.<br/>
    **password**: contains the password to access the email address previously set.<br/>
    **subject**: the subject of the email to be sent.<br/>

### Example of use
```
# With datasets
git clone https://github.com/mhaut/ThuleRSbot/

# configuration file etc/config as explained in the previous section

# Finally launch the bot!
python Main.py
```


## Citing us 

If you use or build on our work, please consider citing us:

```
@misc{redondo2020ThuleRSbot,
    title={ThuleRSbot: Analysis of remote sensing images through social tools},
    author={A. Redondo, J. M. Haut, M. E. Paoletti, J. Plaza and A. Plaza},
    year={2020},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
