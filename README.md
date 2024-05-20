# Exploring Factors Influencing Student Success in Schools

This dataset includes various attributes related to students' personal, social, and academic backgrounds, collected to analyze their impact on academic performance.

## Attributes

1. **school**: Student's school (binary: 'GP' for Gabriel Pereira or 'MS' for Mousinho da Silveira)
2. **sex**: Student's sex (binary: 'F' for female or 'M' for male)
3. **age**: Student's age (numeric: from 15 to 22)
4. **address**: Student's home address type (binary: 'U' for urban or 'R' for rural)
5. **famsize**: Family size (binary: 'LE3' for less than or equal to 3 or 'GT3' for greater than 3)
6. **Pstatus**: Parent's cohabitation status (binary: 'T' for living together or 'A' for apart)
7. **Medu**: Mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education, or 4 – higher education)
8. **Fedu**: Father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education, or 4 – higher education)
9. **Mjob**: Mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
10. **Fjob**: Father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
11. **reason**: Reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
12. **guardian**: Student's guardian (nominal: 'mother', 'father' or 'other')
13. **traveltime**: Home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
14. **studytime**: Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
15. **failures**: Number of past class failures (numeric: n if 1<=n<3, else 4)
16. **schoolsup**: Extra educational support (binary: yes or no)
17. **famsup**: Family educational support (binary: yes or no)
18. **paid**: Extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
19. **activities**: Extra-curricular activities (binary: yes or no)
20. **nursery**: Attended nursery school (binary: yes or no)
21. **higher**: Wants to take higher education (binary: yes or no)
22. **internet**: Internet access at home (binary: yes or no)
23. **romantic**: With a romantic relationship (binary: yes or no)
24. **famrel**: Quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
25. **freetime**: Free time after school (numeric: from 1 - very low to 5 - very high)
26. **goout**: Going out with friends (numeric: from 1 - very low to 5 - very high)
27. **Dalc**: Workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
28. **Walc**: Weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
29. **health**: Current health status (numeric: from 1 - very bad to 5 - very good)
30. **absences**: Number of school absences (numeric: from 0 to 93)

### Target Variable

31. **G1**: First year grade (numeric: from 0 to 20)
32. **G2**: Second year grade (numeric: from 0 to 20)
33. **G3**: Final year (numeric: from 0 to 20, output target)

## Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/mohammed-alaa40123/Exploring-Factors-Influencing-Student-Success-in-Universities.git
    cd Exploring-Factors-Influencing-Student-Success-in-Universities
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit dashboard:
    ```bash
    streamlit run dashboard.py
    ```

### Citation

P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
