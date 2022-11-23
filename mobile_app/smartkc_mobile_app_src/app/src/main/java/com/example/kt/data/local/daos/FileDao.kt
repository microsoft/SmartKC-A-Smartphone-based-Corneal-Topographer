package com.example.kt.data.local.daos

import androidx.room.*
import com.example.kt.data.models.KTFile

@Dao
interface FileDao {
    // Get Un-uploaded files
    @Query("SELECT * FROM files WHERE uploaded = 0") suspend fun getUnUploadedFiles(): List<KTFile>

    // Mark file Uploaded
    @Query("UPDATE files SET uploaded = 1 WHERE uri=:uri") suspend fun markFileUploaded(uri: String)

    // Insert a new file record
    @Query("INSERT INTO files VALUES (:uri, :fileName, 0)") suspend fun insertFileRecord(uri: String, fileName:String)

}