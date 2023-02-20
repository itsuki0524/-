#include <LiquidCrystal.h>
#include <avr/wdt.h>

//LCD表示
LiquidCrystal lcd(9, 8, 4, 5, 6, 7);

//SW0(時間設定)スイッチエッジ検出
int sw0PrevState = 0;  //1つ前のスイッチ状態
int sw0CurrState = 0;  //現在のスイッチ状態
int sw0UpEdge    = 0;  //立ち上がりエッジ検出
int pushsw0Single = 0; //1回押された判定

//SW1(Start)スイッチ関連
int sw1PrevState = 0;   //1つ前のスイッチ状態
int sw1CurrState = 0;   //現在のスイッチ状態
int pushsw1Single = 0;  //1回押された判定

//時間
long timeNow;           //現在時刻
long timeStart;         //タイマー開始時刻
long timeStop;          //一時停止時刻
int timeSet = 0;        //スイッチ設定時刻
int timeDisp = 0;       //表示時刻
int minutes = 0;        //分
int second = 0;         //秒

//制御
int  i;
//タイマー状態 0:初期 1:時間設定 2:カウントダウン 3:カウントアップ 4:一時停止 5:アラーム
int timeState=0;  
void up_time_disp() { 
    if( timeSet > 5939 ) {
        timeSet = 0; 
    }
    else{ 
        timeSet = timeDisp + 60; 
    }
}

//LCDの表示の制御
void lcd_display() {
    lcd.setCursor(0, 0);
    lcd.print(minutes);
    lcd.setCursor(3, 0);
    lcd.print("m");
    lcd.setCursor(5, 0);
    lcd.print(second);
    lcd.setCursor(8, 0);
    lcd.print("s");
    if(minutes<10){
        lcd.setCursor(1, 0);
        lcd.print(" ");
    }
    if(second<10){
        lcd.setCursor(6, 0);
        lcd.print(" ");
    }
    if(timeState==0){
        lcd.setCursor(0, 1);
        lcd.print("Default      ");
    }
    else if(timeState==1){
        lcd.setCursor(0, 1);
        lcd.print("Timeset      ");
    }
    else if(timeState==2){
        lcd.setCursor(0, 1);
        lcd.print("CountDown    ");
    }
    else if(timeState==3){
        lcd.setCursor(0, 1);
        lcd.print("CountUp      ");
    }
    else if(timeState==4){
        lcd.setCursor(0, 1);
        lcd.print("Stop         ");
    }
    else if(timeState==5){
        lcd.setCursor(0, 1);
        lcd.print("Alarm        ");
    }
}

//リセット
void software_reset() {
  asm volatile ("  jmp 0");  
} 

//初期化
void setup() { 
    //入出力の初期化 
    pinMode(A0, INPUT); 
    pinMode(A1, INPUT); 
    pinMode(A5, OUTPUT); 

    //LCDの設定
    lcd.begin(16, 2);
}

//ループ
void loop() {  
    //現在時刻の取得 
    timeNow = millis();  

    //SW0(分ボタン)立ち上がりエッジ検出  
    sw0CurrState = digitalRead(A0); //現在のスイッチ状態     

    //現在のスイッチ状態と一つ前のスイッチ状態を比較 
    if(timeState==0 && sw0CurrState==HIGH && sw0CurrState!=sw0PrevState){ 
        pushsw0Single = 1;
        up_time_disp(); 
    } 

    else if(timeState==1 && sw0CurrState==HIGH && sw0CurrState!=sw0PrevState){
        pushsw0Single = 1;
        up_time_disp();
    }
    else if(timeState==4 && sw0CurrState==HIGH && sw0CurrState!=sw0PrevState){
        pushsw0Single = 1;
        up_time_disp();
    }
    //一つ前のスイッチ状態  
    sw0PrevState = sw0CurrState; 

    //SW1(Startボタン)スイッチ立ち上がりエッジ検出 
    sw1CurrState = digitalRead(A1); //現在のスイッチ状態 
    

    //現在のスイッチ状態と一つ前のスイッチ状態を比較 
    if(timeState==0 && sw1CurrState==HIGH && sw1CurrState!=sw1PrevState){ 
        timeStart = timeNow; 
        pushsw1Single = 1;
    } 
    else if(timeState==1 && sw1CurrState==HIGH && sw1CurrState!=sw1PrevState){
        timeStart = timeNow;
        pushsw1Single = 1;
    }
    else if(timeState==2 && sw1CurrState==HIGH && sw1CurrState!=sw1PrevState){
        timeStop = timeNow;
        pushsw1Single = 1;
    }
    else if(timeState==3 && sw1CurrState==HIGH && sw1CurrState!=sw1PrevState){
        timeStop = timeNow;
        pushsw1Single = 1;
    }
    else if(timeState==4 && sw1CurrState==HIGH && sw1CurrState!=sw1PrevState){
        timeStart = timeNow;
        pushsw1Single = 1;
    }
    else if(timeState==5 && sw1CurrState==HIGH && sw1CurrState!=sw1PrevState){
        pushsw1Single = 1;
      	delay(1000);
    }

    //一つ前のスイッチ状態  
    sw1PrevState = sw1CurrState; 
  
    //同時押しリセット
    if(sw0CurrState==HIGH && sw1CurrState==HIGH){
        software_reset();
    } 


    //動作状態の判定 
      switch(timeState){ 
        //時間設定 
        case 1:   
            //設定された時間を表示
            timeDisp = timeSet;
            if(pushsw0Single == 1){
                timeState = 1;
            }
            else if(pushsw1Single == 1){
                timeState = 2;
            }
          break; 
        //カウントダウン 
        case 2:
            //設定された時間から経過した時間の差を表示
            timeDisp = timeSet - ((timeNow-timeStart)/1000);  
            if(pushsw1Single == 1){ 
                timeState = 1; 
                timeSet = timeDisp;
            } 
            else if(timeDisp == 0){ 
                timeState = 5; 
            } 
          break; 
        //カウントアップ 
        case 3:
            //カウントアップを表示（経過時間）
            timeDisp = timeSet + ((timeNow-timeStart)/1000);
            if(pushsw1Single == 1){
                timeSet = timeDisp;
                timeState = 4;
            }
          break; 
        //一時停止
        case 4:
            //カウントアップを一時停止  
            timeDisp = timeSet;
            if(pushsw0Single == 1){
                timeState = 1;
            }
            else if(pushsw1Single == 1){
                timeState = 3;
            }
          break; 
        //アラーム
        case 5:
            //アラームを鳴らす 
            timeDisp = 0;
            if(pushsw1Single == 1){
                timeState = 0;
                software_reset();
            }
          break; 
        //初期状態 
        case 0:   
            if(pushsw0Single == 1){
                timeState = 1;
            }
            else if(pushsw1Single == 1){
                timeState = 3;
            }
          break; 
    }  

    if(timeState == 5){     
        //動作状態がアラームならば、音を鳴らす 
        tone(A5, 262);
    } 
    else{ 
        //動作状態がアラームでないなら、音を鳴らさない 
        noTone(A5);        
    }
  
    pushsw0Single = 0; 
    pushsw1Single = 0;  
	
    minutes = timeDisp / 60;
    second = timeDisp % 60;
  
	lcd_display();
}