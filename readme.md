# 目录 
- [目录](#目录)
- [leetcode 刷题记录](#leetcode-刷题记录)
  - [1. 两数相加](#1-两数相加)
  - [2. 无重复字符的最长子串](#2-无重复字符的最长子串)
  - [3. 寻找两个正序数组的中位数](#3-寻找两个正序数组的中位数)
  - [4. 最长回文子串](#4-最长回文子串)
  - [5. Z 字形变换](#5-z-字形变换)
  - [6. 整数反转](#6-整数反转)
  - [7. 字符串转换整数 (atoi)](#7-字符串转换整数-atoi)
  - [8. 回文数（简单）](#8-回文数简单)
  - [9. 盛最多水的容器](#9-盛最多水的容器)
  - [10. 整数转罗马数字](#10-整数转罗马数字)
  - [15. 三数之和](#15-三数之和)

# leetcode 刷题记录
When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are 
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
 
The Cauchy-Schwarz Inequality
 
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
## 1. 两数相加
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

我的：
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        timesl1=1
        timesl2=1
        l1_v=0
        l2_v=0
        while True:
            if (not l1) and (not l2):
                break
            if l1:
                l1_v+=l1.val*timesl1
                timesl1*=10
                l1 = l1.next
            if l2:
                l2_v+=l2.val*timesl2
                timesl2*=10
                l2 = l2.next
        result = l1_v + l2_v
        timesl1=1
        a = ListNode(result%10)
        pre = a
        result = int(result/10)
        while result!=0:
            b = ListNode(result%10)
            pre.next = b
            pre = b  
            result = int(result/10)             
        return a
```
牛的
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        res = ListNode()
        dummy = ListNode()
        res.next = dummy
        flag = False
        while l1 or l2 or flag:
            summ = 0
            if l1: summ += l1.val
            if l2: summ += l2.val
            if flag: summ += 1
            if summ >= 10: 
                flag = True
                dummy.val = summ % 10
            else:
                flag = False
                dummy.val = summ
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            if l1 or l2 or flag:
                dummy.next = ListNode()
                dummy = dummy.next
        return res.next

```
主要区别在对于每一位，可以直接使用listnode对应位相加模10，同时以flag存储是否进位不用重构出两个数，并且

## 2. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

我的：
使用```unordered_set```类，注意```.find()```方法是返回一个迭代器，找不到就返回空迭代器，即```.end()```。如果不满足条件（当前元素在子串pre中已经存在了）就将前面的逐个擦除```.erase()```
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int len = 0;
        int result = 0;
        unordered_set<char> pre;
        int idx = 0;
        for (int i = 0; i < s.length(); i++) {
            while (pre.find(s[i]) != pre.end()) {
                pre.erase(s[idx]);
                idx++;
            }
            pre.insert(s[i]);
            len = pre.size();
            if (len > result)
                result = len;
        }
        if (len > result)
            result = len;
        return result;
    }
};
```
佬的：从当前位置i往前看，start记录的是和当前位置i最接近的满足题意的下标：
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
    int max=0,start=0,end=0;
	int n=s.size();
      for(int i=0;i<n;i++)
      {
          end=i;
        for( int j=start;j<i;j++)
	   {
	     if(s[i]==s[j])
		 {
		   start=j+1;
		   max=(max>end-start+1)?max:(end-start+1);
		   break;
		 }
	   }
		   max=(max>end-start+1)?max:(end-start+1);
      }
      return max;
    }
};
```
## 3. 寻找两个正序数组的中位数
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
算法的时间复杂度应该为 O(log (m+n))  (我的好像是O(M+N)？)。

难度标困难但实际比较简单：
```
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1 = (len(nums1)+len(nums2)+1)/2 if (len(nums1)+len(nums2))%2!=0 else (len(nums1)+len(nums2)+1)//2
        idx1,idx2 = 0,0
        flag=0
        for i in range(len1):
            if idx1>=len(nums1) or (idx2<len(nums2) and nums1[idx1]>nums2[idx2]):
                idx2+=1
                flag=2
            else:
                idx1+=1
                flag=1     
        if (len(nums1)+len(nums2))%2!=0:
            if flag==1:
                return nums1[idx1-1]
            else:
                return nums2[idx2-1]
        elif flag==1:
            if idx1>=len(nums1):
                return (nums1[-1]+nums2[idx2])/2.0
            elif idx2>=len(nums2):
                return (nums1[idx1-1]+nums1[idx1])/2.0
            else:
                return (nums1[idx1-1]+min(nums1[idx1],nums2[idx2]))/2.0
        else:
            if idx2>=len(nums2):
                return (nums2[-1]+nums1[idx1])/2.0
            elif idx1>=len(nums1):
                return (nums2[idx2-1]+nums2[idx2])/2.0
            else:
                return (nums2[idx2-1]+min(nums1[idx1],nums2[idx2]))/2.0
```
## 4. 最长回文子串
给你一个字符串 s，找到 s 中最长的回文子串。
示例 2：
```
输入：s = "cbbd"
输出："bb"
```
我的：这题有点坑，主要是对函数不熟悉，```s.substr(i-j, len2)```的参数2应该是长度，一直以为是下标
```
class Solution {
public:
    string longestPalindrome(string s) {
        int maxl = 1;
        string result = "";
        int maxl1 = 0;
        int maxl2 = 0;
        int FLAG = 0;
        int FLAG1 = 1;
        int len = 1;
        int len2 = 0;
        result = s[0];

        for (int i = 0; i < s.length()-1; i++) {
            maxl1 = i < (s.length() - i - 1) ? i : (s.length() - i - 1);
            maxl2 = i < (s.length() - i-2) ? i : (s.length() - i-2);
            FLAG = 0;
            FLAG1 = 1;
            len = 1;
            len2 = 0;
            if (s[i] == s[i + 1]) {
                FLAG = 1;
            }
            for (int j = 0; j <= maxl1; j++) {
                if ((s[i - j] == s[i + j])&& (FLAG1==1) && (j != 0)) {
                    len += 2;
                    if (len > maxl) {
                        result = s.substr(i - j, len);
                        maxl = len;
                    }
                }
                else if((s[i - j] != s[i + j])) FLAG1 = 0;
                if (FLAG == 1) {
                    if (j <=maxl2 && (s[i - j] == s[i + j + 1])) {
                        len2 += 2;
                        if (len2 > maxl) {
                            maxl = len2; 
                            result = s.substr(i-j, len2);
                        }
                    }
                    else if (s[i - j] != s[i + j + 1])FLAG = 0;
                if ((FLAG == 0) && (FLAG1 == 0)) break;   
                }
            }
        }
        return result;
    }
};
```
## 5. Z 字形变换
将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
```
P   A   H   N
A P L S I I G
Y   I   R
```
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
我的思路：根据和行取模，逐项算他的行数，或者是遍历元素时设置一个flag，为1时往下走，即对应行的string加这个字符（后者感觉要快一些）
```
class Solution {
public:
    string convert(string s, int numRows) {
        
        string out[numRows];
        string result="";
        int rows, _div;
        if( numRows==1) return s;
        for (int i = 0; i < s.length(); i++) {
            _div = i % (2 * numRows - 2);
            if (_div < numRows) rows = _div;
            else rows = numRows - (_div - numRows + 1)-1;//计算出当前是第几行的
            out[rows] = out[rows] + s[i];
        }
        for(int i = 0; i < numRows; i++) result+=out[i];
        return result;
    }
};
```
## 6. 整数反转
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
如果反转后整数超过 32 位的有符号整数的范围 [−2^31,  2^31 − 1] ，就返回 0。

感觉对python来说很简单，python无上限好像？难点在判断溢出吧
```
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x<-pow(2,31) or x>(pow(2,31)-1):
            return 0
        times = 1 if x>=0 else -1
        x = abs(x)
        tmp = []
        result=0
        while True:
            if x>=10:
                tmp.append(x%10)
                x=int(x/10)
            else:
                tmp.append(x)
                break
        t = pow(10,len(tmp)-1)# len(tmp)
        for tt in tmp:
            result+=t*tt
            t=t/10
        return times*result if -2147483648 < times*result < 2147483647 else 0
```
针对JAVA 溢出不会报错，可以判断临时的翻转结果，如果这个翻转结果除以10不等于上一个结果，说明有溢出
```
int tmp = res * 10 + x % 10;
if (tmp / 10 != res) { // 溢出!!!
    return 0; 
```
 c++是在中间判断INT_MIN/10和INT_MAX/10的大小关系更大就溢出？应该加个个位数的判断
 ```
 class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            if (rev < INT_MIN / 10 || rev > INT_MAX / 10) {
                return 0;
            }
            int digit = x % 10;
            x /= 10;
            rev = rev * 10 + digit;
        }
        return rev;
    }
};
 ```
## 7. 字符串转换整数 (atoi)
请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
函数 myAtoi(string s) 的算法如下：
读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：
本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符

题目不难 但是考虑的情况不少，很难一次考虑全（符号只能读一次。连续的符号第二个是作为字符了，且首个不能是字母）
```
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        Flag = True
        times=1
        result=0
        tFlag = True
        for s1 in s:
            try:
                #res = int(s1)
                # if s1!=" " and tFlag:
                #     Flag = False
                res = int(s1)

                if result>214748364 or (result==214748364 and res>=8):
                    if times==1:
                        return 2147483647
                    else:
                        return -2147483648 
                result = 10*result+res
                Flag = False
            except:
                if not Flag:
                    return times*result
                if s1=="-" and tFlag:
                    times=-1
                    tFlag = False
                elif s1=="+" and tFlag:
                    tFlag = False
                elif s1==" " and tFlag:
                    continue
                else:
                    return times*result
        return times*result
```
最快的都是正则表达式的（不会） 佬的：
"""
import re
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        INT_MAX = 2147483647    
        INT_MIN = -2147483648
        str = s.lstrip()      #清除左边多余的空格
        num_re = re.compile(r'^[\+\-]?\d+')   #设置正则规则
        num = num_re.findall(str)   #查找匹配的内容
        num = int(*num) #由于返回的是个列表，解包并且转换成整数
        return max(min(num,INT_MAX),INT_MIN)    #返回值
"""
般般快的,思路不难，应该想得到才对。。。
```
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        string=s.strip()
        flag=True
        ans=0
        for index,item in enumerate(string):
            if index==0 and item =='-':
                flag=False
            elif index==0and item =='+':
                flag=True
            elif '9'<item or item < '0':
                break
            else:
                ans=ans*10+(ord(item)-ord('0'))
        ans= ans if flag else -ans
        ans=-2**31 if ans<-2**31 else ans
        ans=2**31-1 if ans>=2**31 else ans
        return ans
```

## 8. 回文数（简单）
给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
```
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        if (x%10 == 0 and x!=0) return false;
        string x1 = to_string(x);
        bool odded = x1.length() % 2 == 0 ? false : true;
        int idx = (x1.length()) / 2;
        if (odded) {
            
            for (int i = 0; i <= idx;i++) {
                if (x1[idx - i] == x1[idx + i]) continue;
                else return false;
            }
        }
        else {
            idx--;
            for (int i = 0; i <= idx; i++) {
                if (x1[idx-i] == x1[idx +1+i]) continue;
                else return false;
            }
        }
        return true;
    
    }
};
```
## 9. 盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
返回容器可以储存的最大水量。

双指针经典，暴力法超时：
```
    int maxArea(vector<int>& height) {
        int max_ = min(height[0], height[1]);
        int area = 0;
        int pre = height[0];

        for (int i = 1; i < height.size(); i++) {
            
            if (abs(height[i] - pre) < 1) continue;
            for (int j = 0; j < i; j++) {
                //h = height[i]<
                if (height[j] < pre) continue;
                pre = min(height[i], height[j]);
                area = pre * (i - j);
                max_ = max(area, max_);
            }
        }
        return max_;
    }
```
```
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                max(res, (j - i) * height[i++]) :
                max(res, (j - i) * height[j--]);
        }
        return res;
    }
};
```
## 10. 整数转罗马数字

最简单的题了应该是
执行用时：4 ms, 在所有 C++ 提交中击败了81.30%的用户
内存消耗：5.7 MB, 在所有 C++ 提交中击败了86.62%的用户

```cpp
class Solution {
public:
    string intToRoman(int num) {
        string a = "";
        int x = 0;
        int i = 0;
        char label[] = { 'I','V','X','L','C','D','M' };
        while (true) {
            string tmp = "";
            x = num % 10;
            if (x < 4) {
                for (int j = 0; j < x; j++) tmp += label[i];
            }else if (x == 4) {
                tmp += label[i];
                tmp += label[i + 1];
            }else if (x == 9) {
                tmp += label[i];
                tmp += label[i + 2];
            }else {
                tmp += label[i + 1];
                for (int j = 0; j < (x - 5); j++) tmp += label[i];
            }
            i += 2;
            num = num / 10;
            a = tmp + a;
            if (num == 0) break;
        }
        return a;
    }
};
```
## 15. 三数之和
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。

难点在要在O($n^2$)解决，以及去重
```
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> result;
        int L;
        int R = nums.size() - 1;
        if (R < 2) return result;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < nums.size(); i++) {
            if(i>0&&nums[i]==nums[i-1]) continue;
            L = i + 1; R = nums.size() - 1;
            while (L < R) {
                if (nums[i] + nums[L] + nums[R] == 0) {
                    result.push_back({ nums[i],nums[L],nums[R] });
                    L++; R--;
                    while (L < R && nums[L] == nums[L - 1]) L++;
                    while (L < R && nums[R] == nums[R + 1]) R--;
                }
                else if (nums[i] + nums[L] + nums[R] < 0) L++;
                else R--;
                
            }
        }
        return result;
    }
};
```
