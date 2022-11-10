package com.example.kt.data.repo

import com.example.kt.data.local.daos.FileDao
import com.example.kt.data.models.KTFile
import javax.inject.Inject

class FileRepository @Inject constructor(
    private val fileDao: FileDao
){
    // insert new file record
    suspend fun insertNewFileRecord(uri: String) { fileDao.insertFileRecord(uri) }
    // get un-uploaded file records
    suspend fun getUnUploadedFileRecords(): List<KTFile> { return fileDao.getUnUploadedFiles() }
}