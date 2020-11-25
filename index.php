
<?php
 $url=$_SERVER['REQUEST_URI'];
 header("Refresh: 5; URL=$url");
 echo "<h2>Total People Crossing Below!</h2>";
 $crossedBelow = file_get_contents('C:\Users\sades\OneDrive\Desktop\crowd-control\crossedBelow.txt');
 echo $crossedBelow;
echo "<h2>Total People Crossing Above!</h2>";
$crossedAbove = file_get_contents('C:\Users\sades\OneDrive\Desktop\crowd-control\crossedAbove.txt');
echo $crossedAbove;

$total = (int)$crossedBelow - (int)$crossedAbove;
#echo $total;
echo "<br>";
if ($total < 3) {
  echo "<h2>Normal Flow!</h2>";
} elseif ($total > 2 and $total < 5) {
  echo "<h2 style=color:yellow>Number of People about to exceed!</h2>";
} else {
  echo "<h2 style=color:red>Number of people Exceeded!</h2>";
}

?>
