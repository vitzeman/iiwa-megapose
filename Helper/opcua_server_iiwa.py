import time
import requests
from opcua import ua, uamethod
from opcua import Server
from datetime import datetime
import queue
import json
import math
from threading import Thread

#### Input Ip here
robot = b'172.31.1.10'

robot_ip = b'http://' + robot + b':30000/'

out_queue = queue.Queue()

class SubHandler(object):
    def datachange_notification(self, node, val, data):
        global out_queue
        # print("Python: New data change event", node, val)
        if val:
            out_queue.put(5)  # TODO: get all parameteres here and put it to queue via dictionary

def check_ready():
    ready_request = robot_ip + b'ready/'
    while requests.get(ready_request).text != "OK":
        # myevgen1.trigger(message='ok')
        print("robot is busy")
        opcua_set_value(inMotion, True)
        # myevgen.trigger(message='busy')
        time.sleep(1.0)

    opcua_set_value(inMotion, False)
    # myevgen.trigger(message='ok')
    return True

def check_fail():
    fail_request = robot_ip + b'failed/'
    response = requests.get(fail_request).text
    print("fail response", response)
    if requests.get(fail_request).text == "OK":
        opcua_set_value(failed, False)
        myevgen1.trigger(message='ok')
        print("Going to the point")
    else:
        opcua_set_value(failed, True)
        myevgen1.trigger(message='failed')
        print("failed to reach the point")

def opcua_set_value(var, value):
    dv = ua.DataValue(value)
    dv.ServerTimestamp = datetime.utcnow()
    dv.SourceTimestamp = dv.ServerTimestamp
    var.set_value(dv)


def init_variable(var, writable=True):
    dv = ua.DataValue(var.get_value())
    dv.ServerTimestamp = datetime.utcnow()
    dv.SourceTimestamp = dv.ServerTimestamp
    var.set_value(dv)
    if writable:
        var.set_writable()

def make_angles_to_range_2pi(angle):
    if angle < 0:
        return 360 + angle
    else:
        return angle

def round_to_degree(data):

    data["A"] = make_angles_to_range_2pi(math.degrees(data["A"]))
    data["B"] = make_angles_to_range_2pi(math.degrees(data["B"]))
    data["C"] = make_angles_to_range_2pi(math.degrees(data["C"]))
    # data["A"] = math.degrees(data["A"])
    # data["B"] = math.degrees(data["B"])
    # data["C"] = math.degrees(data["C"])
    data_round = {key: round(data[key], 3) for key in data}
    return data_round

def update_position_real():
    op = b"Getrealposition"
    req = robot_ip + op
    response = requests.get(req)

    if response.status_code == 200:
        pos = response.content.decode("utf-8")
        pos = pos.replace("\'", "\"")
        data = json.loads(pos)
        data_round = round_to_degree(data)
        print("current position world", data_round)

        opcua_set_value(xA, data_round["x"])
        opcua_set_value(yA, data_round["y"])
        opcua_set_value(zA, data_round["z"])
        opcua_set_value(RAA, data_round["A"])
        opcua_set_value(RBA, data_round["B"])
        opcua_set_value(RCA, data_round["C"])

    else:
        print("Request to getActualPosition was not successful!")

def get_joint_pos():
    op = b"Getjointpostion"
    req = robot_ip + op
    response = requests.get(req)

    if response.status_code == 200:
        pos = response.content.decode("utf-8")
        pos = pos.replace("\'", "\"")
        data = json.loads(pos)

        opcua_set_value(jA1, data["A1"])
        opcua_set_value(jA2, data["A2"])
        opcua_set_value(jA3, data["A3"])
        opcua_set_value(jA4, data["A4"])
        opcua_set_value(jA5, data["A5"])
        opcua_set_value(jA6, data["A6"])
        opcua_set_value(jA7, data["A6"])

    else:
        print("Request to getActualPosition was not successful!")

def update_position_base():

    op = b"Gettransformedposition"
    req = robot_ip + op
    response = requests.get(req)

    if response.status_code == 200:
        pos = response.content.decode("utf-8")
        pos = pos.replace("\'", "\"")
        data = json.loads(pos)
        data_round = round_to_degree(data)
        print("current position base", data_round)

        opcua_set_value(xA, data_round["x"])
        opcua_set_value(yA, data_round["y"])
        opcua_set_value(zA, data_round["z"])
        opcua_set_value(RAA, data_round["A"])
        opcua_set_value(RBA, data_round["B"])
        opcua_set_value(RCA, data_round["C"])

    else:
        print("Request to getActualPosition was not successful!")

def gotobase():

    t_Xval = xT.get_value()
    t_Yval = yT.get_value()
    t_Zval = zT.get_value()
    t_rotAval = RAT.get_value()
    t_rotBval = RBT.get_value()
    t_rotCval = RCT.get_value()

    tra = b'TX=' + str(t_Xval).encode() + b'&TY=' + str(t_Yval).encode() + b'&TZ=' + str(t_Zval).encode()
    rot = b'&TA=' + str(t_rotAval).encode() + b'&TB=' + str(t_rotBval).encode() + b'&TC=' + str(t_rotCval).encode()
    s = tra + rot

    op = b"Gotobase"
    req = robot_ip + op + b'/?' + s
    response = requests.get(req)
    print(response.content)

def gotoworld():

    opcua_set_value(inMotion, True)
    t_Xval = xT.get_value()
    t_Yval = yT.get_value()
    t_Zval = zT.get_value()
    t_rotAval = RAT.get_value()
    t_rotBval = RBT.get_value()
    t_rotCval = RCT.get_value()

    tra = b'TX=' + str(t_Xval).encode() + b'&TY=' + str(t_Yval).encode() + b'&TZ=' + str(t_Zval).encode()
    rot = b'&TA=' + str(t_rotAval).encode() + b'&TB=' + str(t_rotBval).encode() + b'&TC=' + str(t_rotCval).encode()
    s = tra + rot

    op = b"Gotoreal"
    req = robot_ip + op + b'/?' + s
    requests.get(req)

def move_handler():
    try:
        handler = SubHandler()
        sub = server.create_subscription(10, handler)
        handle = sub.subscribe_data_change(ready)
        print("server started")
        update_position_real()

        while True:
            item = out_queue.get()
            opcua_set_value(finished, False)
            opcua_set_value(started, True)
            n = programNumber.get_value()
            print("Execution of program number: " + str(n))
            check_ready()

            if n == 1:
                print("update base position")
                update_position_real()

            elif n == 2:
                print("goto world request")
                gotoworld()

            elif n == 3:
                print("check if last point failed ")
                check_fail()

            else:
                print("Unknown Operation Number")
                continue

            opcua_set_value(finished, True)
            opcua_set_value(started, False)
            print("OPERATION FINISHED")


    finally:
        print("Stopping server")
        server.stop()

if __name__ == "__main__":

    # initialization of Server, its IP and Namespace (URI)
    server = Server()
    server.set_endpoint("opc.tcp://0.0.0.0:5000")
    server.set_server_name("IIWA Control")
    uri = "http://iiwa-control.ciirc.cz"
    idx = server.register_namespace(uri)

    print(f"URI: {uri}, idx: {idx}")
    ns = f"ns={idx}"

    objects = server.get_objects_node()
    # rootobj = objects.add_object(ns+ ";s=IIWA", "IIWA")
    # input_root = rootobj.add_object(ns+ ";s=RobotGenericInput", "RobotGenericInput")
    # output_root = rootobj.add_object(ns+ ";s=RobotGenericOutput", "RobotGenericOutput")
    # InMotion_event = rootobj.add_object(ns+ ";s=InMotionEvent", "InMotionEvent")
    # Fail_event = rootobj.add_object(ns+ ";s=FailEvent", "FailEvent")
    # ready = input_root.add_variable(ns+ ";s=DataReady", "DataReady", ua.Variant(False, ua.VariantType.Boolean))
    # programNumber = input_root.add_variable(ns+ ";s=ProgramNumber", "ProgramNumber", ua.Variant(3, ua.VariantType.Byte))
    rootobj = objects.add_object("ns=2;s=IIWA", "IIWA")
    input_root = rootobj.add_object("ns=2;s=RobotGenericInput", "RobotGenericInput")
    output_root = rootobj.add_object("ns=2;s=RobotGenericOutput", "RobotGenericOutput")
    InMotion_event = rootobj.add_object("ns=2;s=InMotionEvent", "InMotionEvent")
    Fail_event = rootobj.add_object("ns=2;s=FailEvent", "FailEvent")
    ready = input_root.add_variable("ns=2;s=DataReady", "DataReady", ua.Variant(False, ua.VariantType.Boolean))
    programNumber = input_root.add_variable("ns=2;s=ProgramNumber", "ProgramNumber", ua.Variant(3, ua.VariantType.Byte))

    # mobj = rootobj.add_object("ns=2;s=methodobj", "methodobj")
    # mymethod = mobj.add_method(idx, "mymethod", func, [ua.VariantType.Int64], [ua.VariantType.Boolean])

    # actual position and coordinates
    xA = input_root.add_variable("ns=2;s=ActualX", "ActualX", ua.Variant(0, ua.VariantType.Double))
    yA = input_root.add_variable("ns=2;s=ActualY", "ActualY", ua.Variant(0, ua.VariantType.Double))
    zA = input_root.add_variable("ns=2;s=ActualZ", "ActualZ", ua.Variant(0, ua.VariantType.Double))
    RAA = input_root.add_variable("ns=2;s=ActualRotZ", "ActualRA", ua.Variant(0, ua.VariantType.Double))  # A
    RBA = input_root.add_variable("ns=2;s=ActualRotY", "ActualRB", ua.Variant(0, ua.VariantType.Double))  # B
    RCA = input_root.add_variable("ns=2;s=ActualRotX", "ActualRC", ua.Variant(0, ua.VariantType.Double))  # C

    jA1 = input_root.add_variable("ns=2;s=jA1", "jA1", ua.Variant(0, ua.VariantType.Double))
    jA2 = input_root.add_variable("ns=2;s=jA2", "jA2", ua.Variant(0, ua.VariantType.Double))
    jA3 = input_root.add_variable("ns=2;s=jA3", "jA3", ua.Variant(0, ua.VariantType.Double))
    jA4 = input_root.add_variable("ns=2;s=jA4", "jA4", ua.Variant(0, ua.VariantType.Double))
    jA5 = input_root.add_variable("ns=2;s=jA5", "jA5", ua.Variant(0, ua.VariantType.Double))
    jA6 = input_root.add_variable("ns=2;s=jA6", "jA6", ua.Variant(0, ua.VariantType.Double))
    jA7 = input_root.add_variable("ns=2;s=jA7", "jA7", ua.Variant(0, ua.VariantType.Double))
    frame = input_root.add_variable("ns=2;s=ActualFrame", "ActualFrame", ua.Variant(1, ua.VariantType.Byte))

    # target position and coordinates
    xT = input_root.add_variable("ns=2;s=TargetX", "TargetX", ua.Variant(0, ua.VariantType.Double))
    yT = input_root.add_variable("ns=2;s=TargetY", "TargetY", ua.Variant(0, ua.VariantType.Double))
    zT = input_root.add_variable("ns=2;s=TargetZ", "TargetZ", ua.Variant(0, ua.VariantType.Double))
    RAT = input_root.add_variable("ns=2;s=TargetRotZ", "TargetRA", ua.Variant(0, ua.VariantType.Double))  # A
    RBT = input_root.add_variable("ns=2;s=TargetRotY", "TargetRB", ua.Variant(0, ua.VariantType.Double))  # B
    RCT = input_root.add_variable("ns=2;s=TargetRotX", "TargetRC", ua.Variant(0, ua.VariantType.Double))  # C
    tframe = input_root.add_variable("ns=2;s=TargetFrame", "TargetFrame", ua.Variant(0, ua.VariantType.Byte))

    started = input_root.add_variable("ns=2;s=OperationStarted", "OperationStarted", ua.Variant(False, ua.VariantType.Boolean))
    finished = input_root.add_variable("ns=2;s=OperationFinished", "OperationFinished", ua.Variant(True, ua.VariantType.Boolean))
    inMotion = input_root.add_variable("ns=2;s=InMotion", "InMotion", ua.Variant(True, ua.VariantType.Boolean))
    failed = input_root.add_variable("ns=2;s=Failed", "Failed", ua.Variant(True, ua.VariantType.Boolean))
    error = input_root.add_variable("ns=2;s=Error", "Error", ua.Variant(0, ua.VariantType.Int32))
    dataRecency = input_root.add_variable("ns=2;s=DataRecency", "DataRecency",  ua.Variant(False, ua.VariantType.Boolean))

    init_variable(ready)
    init_variable(programNumber)
    init_variable(xA)
    init_variable(yA)
    init_variable(zA)
    init_variable(RAA)
    init_variable(RBA)
    init_variable(RCA)
    init_variable(frame)

    init_variable(xT)
    init_variable(yT)
    init_variable(zT)
    init_variable(RAT)
    init_variable(RBT)
    init_variable(RCT)
    init_variable(tframe)

    init_variable(jA1)
    init_variable(jA2)
    init_variable(jA3)
    init_variable(jA4)
    init_variable(jA5)
    init_variable(jA6)
    init_variable(jA7)

    # output server variables
    init_variable(started, False)
    init_variable(finished, False)
    init_variable(inMotion, False)
    init_variable(failed, True)
    init_variable(error, False)
    init_variable(dataRecency, False)

    etype = server.create_custom_event_type(idx, 'MyFirstEvent', ua.ObjectIds.BaseEventType,
                                            [('MyNumericProperty', ua.VariantType.Float),
                                             ('MyStringProperty', ua.VariantType.String)])

    etype1 = server.create_custom_event_type(idx, 'MyFirstEvent', ua.ObjectIds.BaseEventType,
                                            [('MyNumericProperty', ua.VariantType.Float),
                                             ('MyStringProperty', ua.VariantType.String)])

    myevgen = server.get_event_generator(etype, InMotion_event)
    myevgen1 = server.get_event_generator(etype1, Fail_event )

    server.start()
    # t1 = Thread(target= update_position_real)
    # t2 = Thread(target= update_position_base)
    t3 = Thread(target= move_handler)

    # t1.start()
    # t2.start()
    t3.start()

    # t1.join()
    # t2.join()
    t3.join()



