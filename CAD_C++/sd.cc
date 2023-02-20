#include <LiquidCrystal.h>
#include <avr/wdt.h>

//LCD表示
LiquidCrystal lcd(9, 8, 4, 5, 6, 7);

//SW0(時間設定)スイッチエッジ検出
int sw0PrevState = 0;  //1つ前のスイッチ状態
int sw0CurrState = 0;  //現在のスイッチ状態

//SW1(Start)スイッチ関連
int sw1PrevState = 0;   //1つ前のスイッチ状態
int sw1CurrState = 0;   //現在のスイッチ状態

//同時押し(リセット)関連
int pushwithDouble = 0;  //同時に押された判定 

//時間
long timeNow;           //現在時刻
long timeStart;         //タイマー開始時刻
long timeStop;          //一時停止時刻
int setTime = 0;        //設定時刻(秒単位)
int timeSet = setTime;  //スイッチ設定時刻
int timeDisp = setTime; //表示時刻
int minutes = 0;        //分
int second = 0;         //秒

//制御
int  i;

//タイマー状態 0:初期 1:時間設定 2:カウントダウン 3:カウントアップ 4:一時停止 5:アラーム
int timeState=0;  

void up_time_disp() { 
    if( timeDisp == 6000 ) 
        timeSet = 0; 
    else 
        timeSet += 60; 
}

//リセット
void software_reset() {
  wdt_disable();
  wdt_enable(WDTO_15MS);
  while (1) {}
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

    //SW1(Startボタン)スイッチ立ち上がりエッジ検出 
    sw1CurrState = digitalRead(A1); //現在のスイッチ状態 

    //動作状態の判定 
      switch(timeState){ 
        //時間設定 
        case 1:   
            //設定された時間を表示
        	timeDisp = timeSet;
            if(sw0CurrState==HIGH){
                up_time_disp(); 
                timeState = 1;
            }
            else if(sw1CurrState==HIGH){
                timeStart = timeNow;
                timeState = 2;
            }
          break; 
        //カウントダウン 
        case 2:
            //設定された時間から経過した時間の差を表示
            timeDisp = timeSet -( (timeNow-timeStart)/1000 );  
            if(sw1CurrState==HIGH){
                timeSet = timeDisp;
                timeState = 1; 
            } 
            else if((timeNow-timeStart)/1000>=(timeSet)){ 
                timeState = 5; 
            } 
          break; 
        //カウントアップ 
        case 3:
            //カウントアップを表示（経過時間）
            timeDisp += (timeNow-timeStart)/1000;
            if(sw1CurrState==HIGH){
              	timeStop = timeNow;
                timeState = 4; 
            }
            else if(timeDisp>=6000){
              	timeState = 0;
                break;
            }
          break; 
        //一時停止
        case 4:
            //カウントアップを一時停止
        	timeDisp = (timeStop - timeStart)/1000;
            if(sw0CurrState==HIGH){
              	timeSet = timeDisp;
                up_time_disp(); 
                timeState = 1;
            }
            else if(sw1CurrState==HIGH){
                timeState = 3;
            }
          break; 
        //アラーム
        case 5:
            //アラームを鳴らす 
            timeDisp = 0;
            if(sw1CurrState==HIGH){
                timeState = 0;
            }
          break; 
        //初期状態 
        default:  
        	timeDisp = 0;
        	timeSet = 0;
            if(sw0CurrState==HIGH){
                up_time_disp(); 
                timeState = 1;
            }
            else if(sw1CurrState==HIGH){
                timeStart = timeNow;
                timeState = 3;
            }
          break; 
    }  
  
    if(timeState == 5){     
        //動作状態がアラームならば、音を鳴らす 
        tone(A5, 1500);
    } 
    else{ 
        //動作状態がアラームでないなら、音を鳴らさない 
        noTone(A5);        
    }

    minutes = timeDisp / 60;
    second = timeDisp % 60;

    //LCDの表示の制御 
    lcd.setCursor(0, 0);
  	lcd.print(minutes, "s", second, "m");
}