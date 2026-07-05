import logging

WHISKER_ASCII_BANNER_ART_BORDER = "| " + 110 * "-" + " |"

WHISKER_ASCII_BANNER_ART_MOUSE = """

                             OOOOOOOO                                       OOOOOOOOOOO                       
                           O         OOO                                  OO        OOOOOO                    
                         OOO  OOOO      O                               OO      OOOOO    OOO                  
                       OO         OO     O                             O      OO            OO                
                      OO           OO     O            O             OO      OO              O                
                      O             OO     O           O  OO         O      O                 O               
                     O               OO   OOO OOO    OOOOO  OOO     O      OO                  O              
                     O                O OOO O    OOO            OO OO      O                   OO             
                    OO                O      OOOO                  O      OO O                  O             
                    O                 O    O   OOOOO  O      OOOOO        O  O                  O             
                     O         OOOO   O  OOOOO      O        O   OOO      O                     O             
                     O          O   O OOOO                          O   O O   OOOO O            O             
                     OO          OO   O                     OOOOOOO       O OO  O              O              
                      O              O OOOOOOO            OOO      OO          O     O        OO              
                       O     O      OOO      O           OO          OO       OOOOO          OO               
                        OO      O  OO   OOOOO O          O   OOOOOOO  O   O    O            OO                
                          OO      O O  O   OOOO         OO  O   OOOOO  O  O  O            OO                  
                            OOOOOO  O OOOOOOOO          OO  OOOOOOOOO  O      OO       OOOO                   
                                O  OO OOOOOOO            O  O OOOOO O  O          OOOOOO                      
                              OO   OO  O OOOO OOOOOOOO   O  O  OOO  O  O        O                             
                        OOOOOOO     OOO  OOO             OO   OO OOOOOOOOOOOOOOO OO                           
                 O O                   OO    OOOOO         OO   O                  OO                         
                        O OOO  O          O        OO         O              OOOOO  O                         
                 OOOOOOOO  O  O  OOO     OO         O              OOOOOOOOOOOOOOO OOO                        
           OO             O               OOOO   OOOO                             O  O O   O  OOOOOO          
               OOOOOOOO   O O OOOOOOOO      OO   OO     O     O OOOOOOOOOOOOO OOOOO       OOO      OO         
           O  O  OOO   OO OOO        OO        O               O         OOOO O     OOOO O                    
             O     OO       O OOO     OO       OO            OOO  OOO O       OOOO  O   O                     
              O      OOOOO   O OO        OOOOOO  OOOO     OOOOO            OOOO    O O OO                     
                O     OO     OO             OO       OOOOOOOOO             OO   O OO   O                      
                 OO     OO     OO   O         O     OOOOOO  OO          OOOO   OOO  O  O                      
                   O      OOO    OO     O      OOOOOO O    OO        OOOO    OO     O OO                      
                    OO  OO  OO      OOO         OOO      OO    O OOOOO    OOO       O OO                      
                      OO OO  OO        OOO       OO   OOOO             OO           O OO                      
                      O O      OOOOO      OOOOOOO          OO    OOOOOOO           O  O                       
                         OOOO    O      OO O OO  OOOOOOOOOO  OOOOOOOOOOOOO         O  O                       
                           O OOOO OOOOO O OO  OO  OOOO     OOOO   OOOOOOOOO       O  OO                       
                            O   OO  OOOOOOO O OOO O OOO  OOO   OOOO     OOOO      O  O                        
                           OO    OO   OOOO O   O  OOO  OOOOOO  O   OOO   OOO    O    O                        
                             O O   O    O  OO  O  O O OOO   OOOOOOO  OO  OO OOO     O                         
                              OOOOOOO    OO  OO OO  OO  O        O       OO O  OOOOO                          
                                   O OO O OO  OO O O  O  O        O    OOOOOO  OOO                            
                                     O OO  OOO  OOOOOOOO  OO      OOOOOOOOOOOO                                
                                        OO  OOOO OO    OOOOOOOOOOOOOOOOOO                                     
                                          OO   OOOOOOOOOOOOOOOOOOOO                                           
"""

WHISKER_ASCII_BANNER_ART_TEXT = """
 OOOOO     OOOOO     OOOOO OOOOO     OOOOO  OOOO  OOOOOOOOOOO  OOOOO    OOOOOO OOOOOOOOOOOOOO OOOOOOOOOOOOO   
O    OO   O     O   O    OOO   O     O   O O   O OO            O    O  OO    O O             O            OO  
 O    O  O      OO  O    O O   OO   OO   O O   O O    OOOO   O O    OOO    OO  O    OOOOOOOOOOO   OOOOO    O  
 OO   OO O       O O    O  O   OOOOOOO   O O   O      OOOOOOO  O    O     O    O   OOOOOOOOO OO   OO  OO    O 
  O    OO    O    OO   OO  O             O O   O O         OOO O        OO     O           O OO   OOOOOO   OO 
  OO   OO   OO    O    O   O             O O   O  OOOO       O O          O    O           O OO            O  
   O       OO O       O    O   OOOOOOO   O O   O OOOOOOOOO    OO    OOO    OO  O   OOOOOOOOO OO          OO   
    O      O  OO     OO    O   OO   OO   O O   OO             OO    O OO    OO O             OO   OOOO    OO  
    O     O    O     O     O   O     O   O O   OOO          OO O    O  OO      O             OO        O   OO 
     OOOOOO     OOOOO      OOOOO     OOOOO  OOOO  OOOOOOOOOO   OOOOO     OOOOOOOOOOOOOOOOOOOO OOOOO    OOOOO  
"""

VERSION_NUMBER = "0.8.0"


def log_ascii_banner():
    logging.info(WHISKER_ASCII_BANNER_ART_BORDER)
    for line in WHISKER_ASCII_BANNER_ART_MOUSE.split("\n"):
        line = line.strip("\n")
        if line:
            logging.info(f"| {line} |")
    logging.info(WHISKER_ASCII_BANNER_ART_BORDER)
    for line in WHISKER_ASCII_BANNER_ART_TEXT.split("\n"):
        line = line.strip("\n")
        if line:
            logging.info(f"| {line} |")
    logging.info(WHISKER_ASCII_BANNER_ART_BORDER)
    logging.info(f"Version Number: {VERSION_NUMBER}")
