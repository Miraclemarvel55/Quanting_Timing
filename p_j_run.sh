cd /root/eclipse-workspace/SAP_Python_0
pwd
/usr/bin/python /root/eclipse-workspace/SAP_Python_0/quick_main.py >../p.log 2>&1
cd /root/eclipse-workspace/Report_Html_Generator
pwd
/usr/bin/java  -cp ./bin:./opencsv-4.1.jar:./commons-lang3-3.7.jar start_main.Generator >../j.log
