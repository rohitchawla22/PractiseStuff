package com.company;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        int data[] = {5,6,7,7,8,8,9,10,11,12};
        int nums[] = {2,3,6,7};
        String aa[] = {"a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"};
        String words1[] = {"great","acting","skills"};
        String words2[] = {"fine","drama","talent"};
        String[][] pairs = {{"great","good"},{"fine","good"},{"drama","acting"},{"skills","talent"}};
        boolean isPossible = isPossible(data);
        boolean validWord = validWordAbbreviation("internationalization", "i5a11o1");
        boolean result = areSentencesSimilarTwo(words1, words2, pairs);
        int[] arr = twoSum(nums, 6);
        int a = lengthOfLongestSubstring("dvdf");
        double res = findMedianSortedArrays(data, nums);
        String palindrome = longestPalindrome("ccc");
        String rev = reverseString("helloWorl");
        String reverseWords = reverseWords("the sky is blue");
        int maxSubArrayLen = maxSubArrayLen(nums, 1);
        int versionCompare = compareVersion("01", "1");
        int duplicate = findDuplicate(nums);
        int[] productSum = productExceptSelf(nums);
        String numberToWords = numberToWords(1000);
        int roman = romanToInt("III");
        String intToRoman = intToRoman(27);
        String prefix = longestCommonPrefix(aa);
        String[] renderLogs = reorderLogFiles(aa);
        int uniqueValue = singleNumber(nums);
        char[] chars = {'a','a','b','b','c','c','c'};
        int compress = compress(chars);
        List<String> comb = letterCombinations("23");
        List<List<Integer>> combinationSum = combinationSum(nums, 7);
        System.out.println(numberToWords);
    }
    private static boolean isPossible(int[] nums) {
        // calculates the frequency of all the numbers
        Map<Integer, Integer> frequency = new HashMap<>();
        // final needed array that would produce the consecutive string
        Map<Integer, Integer> needed = new HashMap<>();
        for (int num : nums) {
            frequency.put(num, frequency.getOrDefault(num, 0)+1);
        }
        for (int num : nums) {
            if (frequency.get(num) == 0)
                continue;
            // is it eligible to append to an existing sequence... if its, append it.
            if (needed.containsKey(num -1) && needed.get(num -1) > 0) {
                frequency.put(num, frequency.getOrDefault(num, 0)-1);
                needed.put(num - 1, needed.getOrDefault(num-1, 0)-1);
                needed.put(num, needed.getOrDefault(num, 0)+1);
            // if not, check if there are enough numbers in the sequence... and add the third one to the new sequence
            } else if(frequency.containsKey(num+1)&&
                    frequency.containsKey(num+2)&&
                    frequency.getOrDefault(num+1, 0) > 0 &&
                    frequency.getOrDefault(num+2, 0) > 0
            ) {
                frequency.put(num, frequency.getOrDefault(num, 0)-1);
                frequency.put(num + 1, frequency.getOrDefault(num +1, 0)-1);
                frequency.put(num + 2, frequency.getOrDefault(num +2, 0)-1);
                needed.put(num+2, needed.getOrDefault(num+2, 0)+1);
            } else {
                return false;
            }
        }
        return true;
    }

    private static boolean areSentencesSimilarTwo(String[] words1, String[] words2, String[][] pairs) {
        int len1 = words1.length;
        int len2 = words2.length;
        Map<String, String> pair = new HashMap<>();

        if (len1 != len2) {
            return false;
        }
        for(String[] p : pairs) {
            String parent1 = findMatch(pair, p[0]);
            String parent2 = findMatch(pair, p[1]);
            if (!parent1.equals(parent2)) {
                pair.put(parent1, parent2);
            }
        }

        for (int i = 0; i < len1; i++) {
            if (!findMatch(pair, words1[i]).equals(findMatch(pair, words2[i])) && !words1[i].equals(words2[i])) {
                return false;
            }
        }
        return true;
    }
    private static String findMatch(Map<String, String> m, String s) {
        if (!m.containsKey(s)) m.put(s, s);
        return s.equals(m.get(s)) ? s : findMatch(m, m.get(s));
    }
    // TWO SUMS
    private static int[] twoSum(int[] nums, int target) {
        int[] finalArray = new int[2];
        for (int i = 0; i < nums.length; i++)
            for (int j = i+1; j < nums.length; j++)
                if (nums[i] + nums[j] == target) {
                    finalArray[0] = i;
                    finalArray[1] = j;
                    return finalArray;
                }
        return finalArray;
    }
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode check = new ListNode(0);
        ListNode head = check;
        int carry= 0;
        while (l1 != null || l2 !=null) {
            int sum = carry;
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            carry = sum / 10;
            sum = sum % 10;
            head.next = new ListNode(sum);
            head = head.next;
        }
        if (carry > 0) {
            head.next = new ListNode(carry);
        }
        return  check.next;
    }

    private static int lengthOfLongestSubstring(String s) {
        int start = 0; int current = 0; int result = 0;
        HashSet set = new HashSet();
        while (current < s.length()) {
            if (!set.contains(s.charAt(current))) {
                set.add(s.charAt(current));
                current++;
                result = Math.max(result, set.size());
            } else {
                set.remove(s.charAt(start));
                start++;
            }
        }
        return result;
    }

    private static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int[] temp = new int[nums1.length + nums2.length];
        int totalLenth = temp.length;
        System.arraycopy(nums1, 0, temp, 0, nums1.length);
        System.arraycopy(nums2, 0, temp, nums1.length, nums2.length);
        Arrays.sort(temp);
        return totalLenth % 2 == 1 ? (double)temp[(totalLenth-1)/2] : (double)(temp[(totalLenth-1)/2] + temp[(totalLenth-1)/2 + 1])/2;
    }
    private static String palindromeHelper(String s, int start, int end) {

        while (start >=0 && end < s.length() && s.charAt(start) == s.charAt(end)) {
            start--;
            end++;
        }
        return s.substring(start+1, end);
    }
    private static String longestPalindrome(String s) {
        String longestPanlindromeTillNow = s.substring(0, 1);
        for(int i =0; i < s.length()-1; i++) {
            String temp = palindromeHelper(s, i, i);
            if (temp.length() > longestPanlindromeTillNow.length()) {
                longestPanlindromeTillNow = temp;
            }
            temp = palindromeHelper(s, i, i+1);
            if (temp.length() > longestPanlindromeTillNow.length()) {
                longestPanlindromeTillNow = temp;
            }
        }
        return longestPanlindromeTillNow;
    }

    /**
     * Reverse String
     * Write a function that reverses a string. The input string is given as an array of characters char[].
     *
     * Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
     * @param str
     */
    public static String reverseString(String str) {
        char[] s = str.toCharArray();
        int length = s.length;
        int half = (int) Math.floor(length/2);
        for (int i =0; i<half; i++) {
            s[i] ^= s[length - i - 1];
            s[length - i - 1] ^= s[i];
            s[i] ^= s[length - i - 1];
        }
        return String.valueOf(s);
    }

    /**
     * Given an input string , reverse the string word by word.
     * A word is defined as a sequence of non-space characters.
     * The input string does not contain leading or trailing spaces.
     * The words are always separated by a single space.
     * Follow up: Could you do it in-place without allocating extra space?
     * @param input
     */
    public static String reverseWords(String input) {
        char[] str = input.toCharArray();
        int length = str.length;
        int position = 0;
        for (int i = 0; i <= length; i++) {
            if (i< length && str[i] != ' ') {
                continue;
            }
            reverse(str, position, i-1);
            position =i+1;
        }
        reverse(str, 0, length-1);
        return String.valueOf(str);
    }
    private static void reverse(char[] rev, int start, int end) {
        while (start < end) {
            char temp = rev[start];
            rev[start] = rev[end];
            rev[end] = temp;
            start++;
            end--;
        }
    }

    public static int maxSubArrayLen(int[] nums, int k) {
        int length = nums.length;
        int sum = 0;
        int maxLen = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i<length; i++) {
            sum += nums[i];
            if (!map.containsKey(sum)) {
                map.put(sum, i);
            }
            if (map.containsKey(sum - k)) {
                int pos = map.get(sum - k);
                maxLen = Math.max(maxLen, i - pos);
            }
        }
        return maxLen;
    }

    public static int compareVersion(String s1, String s2) {
        int i =0;
        int k=0;
        int num1 = 0;
        int num2 = 0;
        while (i < s1.length() || k < s2.length()) {
            while (i < s1.length() && s1.charAt(i) != '.') {
                num1 = num1 *10 + s1.charAt(i) - '0';
                i++;
            }
            while (k < s2.length() && s2.charAt(k) != '.') {
                num2 = num2*10 + s2.charAt(k) - '0';
                k++;
            }
            if (num1 > num2) {
                return 1;
            }
            if (num1 < num2) {
                return -1;
            }
            num1 = num2 = 0;
            i++;
            k++;
        }
        return 0;
    }
    public static int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] sumArray = new int[length];
        int[] left = new int[length];
        int[] right= new int[length];
        left[0] = 1;
        right[length-1] = 1;
        for (int i = 1; i < length; i++) {
            left[i] = left[i-1]*nums[i-1];
        }
        for (int i = length-2; i>=0; i--) {
            right[i] = right[i+1]*nums[i+1];
        }
        for (int i = 0; i < length; i++) {
            sumArray[i] = left[i]*right[i];
        }
        return sumArray;
    }

    /**
     * Integer to English words
     * Convert a non-negative integer to its english words representation. Given input is guaranteed to be less than 231
     * @param num
     * @return
     */
    public static String numberToWords(int num) {
        Map<Integer, String> map = new HashMap<>();
        map.put(0, "Zero");
        map.put(1, "One");
        map.put(2, "Two");
        map.put(3, "Three");
        map.put(4, "Four");
        map.put(5, "Five");
        map.put(6, "Six");
        map.put(7, "Seven");
        map.put(8, "Eight");
        map.put(9, "Nine");
        map.put(10, "Ten");
        map.put(11, "Eleven");
        map.put(12, "Twelve");
        map.put(13, "Thirteen");
        map.put(14, "Fourteen");
        map.put(15, "Fifteen");
        map.put(16, "Sixteen");
        map.put(17, "Seventeen");
        map.put(18, "Eighteen");
        map.put(19, "Nineteen");
        map.put(20, "Twenty");
        map.put(30, "Thirty");
        map.put(40, "Forty");
        map.put(50, "Fifty");
        map.put(60, "Sixty");
        map.put(70, "Seventy");
        map.put(80, "Eighty");
        map.put(90, "Ninety");
        StringBuilder sb = new StringBuilder();
        if (num >= 1000000000) {
            int extraNum = num/1000000000;
            sb.append(extra(extraNum, map)+ " Billion");
            num = num%1000000000;
        }
        if (num >= 1000000) {
            int extraNum = num/1000000;
            sb.append(extra(extraNum, map)+ " Million");
            num = num%1000000;

        }
        if (num >= 1000) {
            int extraNum = num/1000;
            sb.append(extra(extraNum, map)+ " Thousand");
            num = num%1000;
        }
        if (num > 0) {
            sb.append(extra(num, map));
        }
        return sb.toString();
    }
    private static String extra(int num, Map<Integer, String> map) {
        StringBuilder sb = new StringBuilder();
        if(num>=100){
            int numHundred = num/100;
            sb.append(" " +map.get(numHundred)+ " Hundred");
            num = num%100;
        }
        if (num > 0){
            if (num >= 0 && num < 20) {
                sb.append(" "+map.get(num));
            } else {
                int extra = num/10;
                sb.append(" "+map.get(extra*10));
                int singleDigit = num%10;
                if (singleDigit > 0) {
                    sb.append(" "+map.get(singleDigit));
                }
            }
        }
        return sb.toString();
    }

    /**
     * Find the Duplicate Number
     * Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.
     * @param nums
     * @return
     */
    public static int findDuplicate(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i< nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0)+1);
        }
        for (int num : nums) {
            if (map.get(num) > 1) {
                return num;
            }
        }
        return 0;
    }

    /**
     * Intersection of Two Linked Lists
     * Write a program to find the node at which the intersection of two singly linked lists begins.
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        Set set = new HashSet();
        while (headA != null) {
            set.add(headA);
            headA = headA.next;
        }
        while (headB != null) {
            if (set.contains(headB)) {
                return headB;
            }
            headB = headB.next;
        }
        return null;
    }

    /**
     *   Merge k Sorted Lists
     * Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }

        PriorityQueue<ListNode> pq = new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        ListNode head = new ListNode(0);
        ListNode p = head;
        for (ListNode list : lists) {
            if (list != null) {
                pq.offer(list);
            }
        }
        while (!pq.isEmpty()) {
            ListNode ln = pq.poll();
            p.next = ln;
            p = p.next;

            if (ln.next!=null) {
                pq.offer(ln.next);
            }
        }
        return head.next;
    }

    /**
     * Validate BST
     * Given a binary tree, determine if it is a valid binary search tree (BST).
     *
     * Assume a BST is defined as follows:
     *
     * The left subtree of a node contains only nodes with keys less than the node's key.
     * The right subtree of a node contains only nodes with keys greater than the node's key.
     * Both the left and right subtrees must also be binary search trees.
     * @param root
     * @return
     */
    public static boolean isValidBST(TreeNode root) {
        if (root == null)    return true;

        boolean left = (root.left != null) ? checkValidity(root.left, Long.MIN_VALUE, root.val) : true;
        boolean right = (root.right != null) ? checkValidity(root.right, root.val, Long.MAX_VALUE) : true;

        return left && right;
    }

    private static boolean checkValidity(TreeNode root, long lo, long hi) {
        if (!(root.val < hi && root.val > lo))    return false;

        // Traverse to left and right children
        boolean left = (root.left != null) ? checkValidity(root.left, lo, root.val) : true;
        boolean right = (root.right != null) ? checkValidity(root.right, root.val, hi) : true;

        return left && right;
    }

    public static boolean isSymmetric(TreeNode root) {
        boolean treeSymmetric = isTreeSymmetric(root, root);
        return treeSymmetric;
    }

    private static boolean isTreeSymmetric(TreeNode root, TreeNode root1) {
        if (root == null && root1 == null) {
            return true;
        }
        if (root == null || root1 == null) {
            return false;
        }
        if (root.val == root1.val) {
            if(isTreeSymmetric(root.left, root1.right)){
                return isTreeSymmetric(root.right, root1.left);
            }
        }
        return false;
    }

    /**
     * Closest Binary Search Tree Value
     * Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.
     *
     * Note:
     *
     * Given target value is a floating point.
     * You are guaranteed to have only one unique value in the BST that is closest to the target.
     * @param root
     * @param target
     * @return
     */
    public int closestValue(TreeNode root, double target) {
        int result = root.val;
        if (target < result && root.left !=null) {
            int leftResult = closestValue(root.left, target);
            if (Math.abs(result - target) >= Math.abs(leftResult - target)){
                result = leftResult;
            }
        } else if (target > result && root.right !=null) {
            int rightResult = closestValue(root.right, target);
            if (Math.abs(result - target) >= Math.abs(rightResult - target)) {
                result = rightResult;
            }
        }
        return result;
    }

    /**
     * 13. Roman to Integer
     * Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
     * @param s
     * @return
     */
    public static int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);

        char[] c = s.toCharArray();
        int sum = map.get(s.charAt(s.length()-1));
        for ( int i = c.length-2; i >= 0; i--) {
            if (map.get(c[i])>= map.get(c[i+1])) {
                sum = sum+map.get(c[i]);
            } else {
                sum = sum - map.get(c[i]);
            }
        }
        return sum;
    }
    public static String intToRoman(int number) {
        String eqv = "";
        while (number>=1000) {
            eqv = eqv + "M";
            number = number - 1000;
        }
        while (number >= 900) {
            eqv += "CM";
            number = number - 900;

        }
        while (number >= 500) {
            eqv += "D";
            number = number - 500;
        }

        while (number >= 400) {
            eqv += "CD";
            number = number - 400;
        }
        while (number >= 100) {
            eqv += "C";
            number = number - 100;
        }
        while (number >= 90) {
            eqv += "XC";
            number = number - 90;
        }
        while (number >= 50) {
            eqv += "L";
            number = number - 50;
        }
        while (number >= 40) {
            eqv += "XL";
            number = number - 40;
        }
        while (number >= 10) {
            eqv += "X";
            number = number - 10;
        }
        while (number >= 9) {
            eqv += "IX";
            number = number - 9;
        }
        while (number >= 5) {
            eqv += "V";
            number = number - 5;
        }
        while (number >= 1) {
            eqv += "I";
            number = number - 1;
        }
        return eqv;
    }

    public static String longestCommonPrefix(String[] strs) {
        String longestPrefix = "";
        StringBuilder db = new StringBuilder();
        Arrays.sort(strs);
        String head = strs[0];
        String tail = strs[strs.length - 1];
        if (head.isEmpty()) {
            return longestPrefix;
        }
        int size = head.length() < tail.length() ? head.length() : tail.length();
        for (int j = 0; j < size; j++) {
            if (head.charAt(j) != tail.charAt(j)) {
                break;
            }
            db.append(head.charAt(j));
        }

        return db.toString();
    }

    /**
     * Reorder Log Files
     * You have an array of logs.  Each log is a space delimited string of words.
     *
     * For each log, the first word in each log is an alphanumeric identifier.  Then, either:
     *
     * Each word after the identifier will consist only of lowercase letters, or;
     * Each word after the identifier will consist only of digits.
     * We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.
     *
     * Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.
     *
     * Return the final order of the logs.
     *
     *
     *
     * Example 1:
     *
     * Input: ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
     * Output: ["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]
     * @param logs
     * @return
     */
    public static String[] reorderLogFiles(String[] logs) {
        Arrays.sort(logs, (log1, log2) -> {
            String[] str1 = log1.split(" ", 2);
            String[] str2 = log2.split(" ", 2);
            boolean isLog1Digit = Character.isDigit(str1[1].charAt(0));
            boolean isLog2Digit = Character.isDigit(str2[1].charAt(0));
            if (!isLog1Digit && !isLog2Digit) {
                int comparison = str1[1].compareTo(str2[1]);
                if(comparison != 0) {
                    return comparison;
                }
                return str1[0].compareTo(str2[0]);
            }
            return isLog1Digit ? (isLog2Digit ? 0 : 1): -1;
        });
        return logs;
    }

    /**
     * Sum Root to Leaf Numbers
     *   Go to Discuss
     * Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
     *
     * An example is the root-to-leaf path 1->2->3 which represents the number 123.
     *
     * Find the total sum of all root-to-leaf numbers.
     *
     * Note: A leaf is a node with no children.
     * @param root
     * @return
     */
    public static int sumNumbers(TreeNode root) {
        /**
         * First Number is the root. And as nothing is above the root, we pass value as 0
         *
         */
        return findSum(root, 0);
    }

    private static int findSum(TreeNode root, int value) {
        if (root == null) {
            return 0;
        }
        value = value *10 + root.val;
        // if current node is leaf, return the current value of val
        if (root.left == null && root.right == null) {
            return value;
        }
        return findSum(root.left, value) + findSum(root.right, value);
    }

    private static TreeNode maxBinaryTree(int[] nums, int leftIndex, int rightIndex) {
        if (leftIndex == rightIndex) {
            return null;
        }
        int maxIndex = maxIndex(nums, leftIndex, rightIndex);
        TreeNode root = new TreeNode(nums[maxIndex]);
        root.left = maxBinaryTree(nums, leftIndex, maxIndex-1);
        root.right = maxBinaryTree(nums, maxIndex+1, rightIndex);
        return root;
    }

    private static int maxIndex(int[] arr, int left, int right) {
        int maxIndex = left;
        for (int i = left; i < right; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static int singleNumber(int[] nums) {
        if (nums == null) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums) {
            map.put(i, map.getOrDefault(i, 0)+1);
        }
        int unique = 0;
        for (int i: nums) {
            if (map.get(i) == 1) {
                unique = i;
            }
        }
        return unique;
    }
    private static int singleNum(int[] nums) {
        int res=0;
        for(int num:nums)
        {
            res=res^num;
        }
        return res;

    }

    /**
     * Two Sum IV - Input is a BST
     *
     * Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that
     * their sum is equal to the given target
     * @param root
     * @param k
     * @return
     */
    Set<Integer> set = new HashSet<>();
    public boolean findTargetWithSet(TreeNode root, int k) {
        /**
         * there are 2 ways to solve this problem
         * 1. Use Inorder traversal -> convert to array, traverse array and find if k is present with 2 sum.
         * 2. Use the SET and save the the diff of the root value in set. check if the diff is present in the set when traversing with
         * root.left and root.right
         */
        if (root == null) {
            return false;
        }
        if (findTargetWithSet(root.left, k)) {
            return true;
        }
        if (set.contains(k - root.val)) {
            return true;
        }
        set.add(k-root.val);
        if (findTargetWithSet(root.right, k)) return true;
        return false;
    }
    private static void inorder(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        inorder(root.left, list);
        list.add(root.val);
        inorder(root.right, list);
    }
    public boolean findTargetWIthTreeTraversal(TreeNode root, int k) {
        List<Integer> inOrderList = new ArrayList<>();
        inorder(root, inOrderList);
        int left = 0;
        int right = inOrderList.size() -1;
        while (left < right) {
            if (inOrderList.get(left) + inOrderList.get(right) == k) {
                return true;
            } else if (inOrderList.get(left) + inOrderList.get(right) < k) {
                left++;
            } else {
                right++;
            }
        }
        if (inOrderList.contains(k)) {
            return true;
        }
        return false;
    }

    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null) {
            return false;
        }
        if (t == null) {
            return false;
        }
        if (isIdentical(t, s)) {
            return true;
        }
        return isIdentical(t.left, s) || isIdentical(t.right, s);
    }

    public boolean isIdentical(TreeNode tree, TreeNode subTree) {
        if (subTree == null && tree == null) {
            return true;
        }
        if (subTree == null || tree == null) {
            return false;
        }
        return (
                subTree.val == tree.val
                && isIdentical(tree.left, subTree.left)
                && isIdentical(tree.right, subTree.right)
        );
    }

    public static List<String> letterCombinations(String digits) {
        StringBuilder sb = new StringBuilder();
        List<String> result = new ArrayList<>();
        if (digits.equals("") || digits == null) {
            return result;
        }
        Map<Character, char[]> map = new HashMap<>();
        map.put('0', new char[]{});
        map.put('1', new char[]{});
        map.put('2', new char[]{'a', 'b', 'c'});
        map.put('3', new char[]{'d', 'e', 'f'});
        map.put('4', new char[]{'g', 'h', 'i'});
        map.put('5', new char[]{'j', 'k', 'l'});
        map.put('6', new char[]{'m', 'n', 'o'});
        map.put('7', new char[]{'p', 'q', 'r', 's'});
        map.put('8', new char[]{'t', 'u', 'v'});
        map.put('9', new char[]{'w', 'x', 'y', 'z'});

        combinationHelper(digits, sb, map, result);
        return result;
    }

    private static void combinationHelper(String digits, StringBuilder sb, Map<Character,char[]> map, List<String> result) {
        if (sb.length() == digits.length()) {
            result.add(sb.toString());
            return;
        }
        for (char ch : map.get(digits.charAt(sb.length()))) {
            sb.append(ch);
            combinationHelper(digits, sb, map, result);
            sb.deleteCharAt(sb.length()-1);
        }
    }

    public static boolean validWordAbbreviation(String word, String abbr) {
        int wordLength = word.length();
        int ab = abbr.length();
        int wordCount = 0;
        int count = 0;
        while (wordCount < wordLength && count < ab) {
            if (!Character.isDigit(abbr.charAt(count))) {
                if (word.charAt(wordCount) == abbr.charAt(count)) {
                    wordCount++;
                    count++;
                } else {
                    return false;
                }
            } else {
                if (abbr.charAt(count) == '0') {
                    return false;
                }
                int num = 0;
                while (count < abbr.length()  && Character.isDigit(abbr.charAt(count)) ) {
                    int digit = Character.getNumericValue(abbr.charAt(count));
                    num = num*10 + digit;
                    count++;
                }
                wordCount+=num;
            }
        }
        return  wordCount == wordLength && count == ab;
    }

    /**
     * Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.
     *
     * The same repeated number may be chosen from candidates unlimited number of times.
     *
     * Note:
     *
     * All numbers (including target) will be positive integers.
     * The solution set must not contain duplicate combinations.
     * Example 1:
     *
     * Input: candidates = [2,3,6,7], target = 7,
     * A solution set is:
     * [
     *   [7],
     *   [2,2,3]
     * ]
     * Example 2:
     *
     * Input: candidates = [2,3,5], target = 8,
     * A solution set is:
     * [
     *   [2,2,2,2],
     *   [2,3,3],
     *   [3,5]
     * ]
     * @param candidates
     * @param target
     * @return
     */
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<Integer> temp = new ArrayList<>();
        List<List<Integer>> result = new ArrayList<>();
        combinationSumHelper(candidates, 0, target, 0, temp, result);
        return result;
    }

    private static void combinationSumHelper(int[] candidates, int start, int target, int sum, List<Integer> temp, List<List<Integer>> result) {
        if (sum > target) {
            return;
        }
        if (sum == target) {
            result.add(new ArrayList<>(temp));
        }
        for (int i = 0; i < candidates.length; i++) {
            temp.add(candidates[i]);
            sum+=candidates[i];
            combinationSumHelper(candidates, i, target, sum, temp, result);
            temp.remove(temp.size()-1);
        }
    }

    public static int compress(char[] chars) {
        StringBuilder sb = new StringBuilder();
        int count = 1;
        int prev = chars[0];
        for (int i = 1; i < chars.length; i++) {
            char current = chars[i];
            if (current == prev) {
                count+=1;
            } else {
                sb.append(prev);
                sb.append((char) (count+'0'));
                chars[++i] = (char)(count+'0');
                count = 1;
            }
            prev = current;
        }
        return sb.toString().length();
    }
    
    public static int pivotIndex(int[] nums) {
        if (nums.length == 0) {
            return -1;
        }
        int pivotIndex = -1;
        int total = 0;
        int sum = 0;
        for (int num : nums) {
            total += num;
        }
        for(int i = 0; i < nums.length; sum += nums[i++]) {
            if(sum * 2 == total - nums[i]) {
                pivotIndex = i;
            }
        }
        return pivotIndex;
//        int i = 0;
//        int j = nums.length-1;
//        int sumLeft = nums[i];
//        int sumRight = nums[j];
//        while(j >= i) {
//            if (sumLeft <= sumRight) {
//                sumLeft += nums[++i];
//            } else {
//                sumRight += nums[--j];
//            }
//            if(sumLeft == sumRight) {
//                pivotIndex = i;
//            }
//        }
//        return pivotIndex;
    }

    public static int dominantIndex(int[] nums) {
        int max = Integer.MIN_VALUE;
        int maxIndex = -1;
        for(int i = 0; i < nums.length; i++) {
            if (max < Math.max(max, nums[i])) {
                max = Math.max(max, nums[i]);
                maxIndex = i;
            }
        }
        for(int i = 0; i < nums.length; i++) {
            if(nums[i]*2 <= max && nums[i]!=max) {
                continue;
            } else {
                if (nums[i] == max) {
                    continue;
                } else {
                    maxIndex = -1;
                }
            }

        }
        return maxIndex;
    }
    public static int[] plusOne(int[] digits) {
        int[] result = new int[digits.length];
        int num;
        for(int i = digits.length-1; i>=0; --i) {
            digits[i] += 1;
            if(digits[i] != 10){
                return digits;
            } else  {
                digits[i] = 0;
                if (i == 0) {
                    int[] newNumber = new int[digits.length+1];
                    newNumber[0] = 1;
                    for (int k = 0; k < digits.length; k++) {
                        newNumber[k+1] = digits[k];
                    }
                    return newNumber;
                }
            }
        }
        return digits;
    }

    public static int[] findDiagonalOrder(int[][] matrix) {
        if (matrix.length == 0) return new int[0];
        int r = 0, c = 0, m = matrix.length, n = matrix[0].length, arr[] = new int[m * n];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = matrix[r][c];
            if ((r + c) % 2 == 0) { // moving up
                if      (c == n - 1) { r++; }
                else if (r == 0)     { c++; }
                else            { r--; c++; }
            } else {                // moving down
                if      (r == m - 1) { c++; }
                else if (c == 0)     { r++; }
                else            { r++; c--; }
            }
        }
        return arr;
    }

    private static int[] sprialMatrix(int[][] matrix) {
        int m = matrix.length;
        int x = 0; int y = 0;
        List<Integer> result = new ArrayList<>();
        if(matrix == null || m == 0) {
            return new int[0];
        }
        int n = matrix[0].length; // x -> row length

        while(m > 0 && n > 0) {
            // check if the matrix has only 1 row/column, then circle cannot be formed so just return that row.
            if(m == 1) {
                for(int i = 0; i < n; i++) {
                    result.add(matrix[x][y++]);
                }
                break;
            } else if(n == 1) {
                for(int i = 0; i < m; i++) {
                    result.add(matrix[x++][y]);
                }
                break;
            }
            //traverse from top -> right
            for(int i = 0; i<n-1; i++) {
                result.add(matrix[x][y++]);
            }
            //traverse from right -> down
            for(int i = 0; i<m-1; i++) {
                result.add(matrix[x++][y]);
            }
            //traverse from down -> left
            for(int i = 0; i<n-1; i++) {
                result.add(matrix[x][y--]);
            }
            //traverse from down -> right
            for(int i = 0; i<m-1; i++) {
                result.add(matrix[x--][y]);
            }
            x++; y++; m-=2; n-=2;
        }
        int[] arr = new int[result.size()];
        for (int i = 0; i<result.size(); i++) {
            arr[i] = result.get( i );
        }
        return arr;
    }

    private static ArrayList<ArrayList<Integer>> generate(int numRows) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        ArrayList<Integer> preResult = new ArrayList<>();

        if(numRows <= 0) {
            return result;
        }
        preResult.add(1);
        result.add( preResult );
        for (int i = 2; i <= numRows; i++) {
            ArrayList<Integer> current = new ArrayList<>(  );
            current.add( 1 );
            for (int j = 0; j < preResult.size()-1; j++){
                current.add( preResult.get(j) + preResult.get(j+1));
            }
            current.add( 1 );
            result.add( current );
            preResult = current;
        }
        return result;
    }

    public static String addBinary(String a, String b) {
        // Noob way
        int num1 = Integer.parseInt( a, 2 );
        int num2 = Integer.parseInt( b, 2 );
        int result = num1+num2;
//        return Integer.toBinaryString( sum );
        StringBuilder sb = new StringBuilder();
        int i = a.length()-1; int j = b.length()-1; int carry = 0;
        while (i >=0 || j >= 0) {
            int sum = carry;
            if (j >= 0 ) {
                sum+= b.charAt(j--) - '0';
            }
            if (i >= 0 ) {
                sum+= a.charAt(i--) - '0';
            }
            sb.append( sum%2);
            carry = sum/2;
        }
        if (carry!=0) {
            sb.append( carry );
        }
        return sb.reverse().toString();
    }

//    private static int strStr(String haystack, String needle) {
//
//    }

    private static int[] kmpArray(String needle) {
        int len = needle.length();
        int[] value = new int[len];
        value[0] = 0;
        for (int i = 1; i < len; i++) {
            int index = value[i-1];
            while (index > 0 && needle.charAt( index ) != needle.charAt( i )) {
                index = value[index-1];
            }
            if (needle.charAt( index ) == needle.charAt( i )) {
                value[i] = value[i-1]+1;
            } else {
                value[i] = 0;
            }
        }
        return value;
    }

    public static int arrayPairSum(int[] nums) {
        Arrays.sort(nums);
        int len = nums.length;
        int i = 1;
        int sum = 0;
        while (i < len ) {
            sum += Math.min( nums[i-1], nums[i] );
            i = i+2;
        }
        return sum;
    }

    public static int findMaxConsecutiveOnes(int[] nums) {
        int maxConsec = 0;
        int finalMax = 0;
        for(int i = 0; i < nums.length; i++) {
            if (nums[i] == 1) {
                maxConsec++;
            } else {
                finalMax = Math.max(finalMax, maxConsec);
                maxConsec = 0;
                continue;
            }
            finalMax = Math.max(finalMax, maxConsec);

        }
        return finalMax;
    }
}

